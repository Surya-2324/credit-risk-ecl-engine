import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.optimize import brentq
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score, roc_curve

os.makedirs("reports/figures", exist_ok=True)

# =====================================================================
# PART 1 – Survival model (Lesson 14): fit on train, save
# =====================================================================
train = pd.read_parquet("data/processed/train.parquet")

last = pd.to_datetime(train["last_pymnt_d"], format="%b-%Y", errors="coerce")
months_paid = ((last.dt.year - train["issue_d"].dt.year) * 12
               + (last.dt.month - train["issue_d"].dt.month)).fillna(0)
train["event"] = train["default"]
train["duration"] = np.where(train["event"] == 1, months_paid + 1, months_paid).clip(1, 36)

COX_FEATS = ["fico", "dti", "annual_inc", "revol_util", "inq_last_6mths",
             "loan_to_income", "mort_acc", "emp_years", "credit_history_years"]
Xc = train[COX_FEATS].astype(float).copy()
Xc["annual_inc"] = np.log1p(Xc["annual_inc"])
median = Xc.median()
Xc = Xc.fillna(median)
stats = {"median": median, "mean": Xc.mean(), "std": Xc.std()}
Xc = (Xc - stats["mean"]) / stats["std"]

cox_df = Xc.assign(duration=train["duration"].values, event=train["event"].values)
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col="duration", event_col="event")
print("Cox hazard ratios (per 1 standard deviation):")
print(cph.summary[["exp(coef)", "p"]].round(3))

kmf = KaplanMeierFitter().fit(train["duration"], train["event"])
et = kmf.event_table
hazard = (et["observed"] / et["at_risk"]).loc[1:36]

surv = {"cox": cph, "stats": stats, "features": COX_FEATS, "km_hazard": hazard}
with open("data/processed/survival_model.pkl", "wb") as f:
    pickle.dump(surv, f)
print("Saved survival model.\n")

# =====================================================================
# PART 2 – Lesson 15, Step 1: load test set and the other models
# =====================================================================
test = pd.read_parquet("data/processed/test.parquet")
test_w = pd.read_parquet("data/processed/test_woe.parquet")

def load(name):
    with open(f"data/processed/{name}.pkl", "rb") as f:
        return pickle.load(f)

lg_pd = load("lgbm_model")
cal = load("calibrator")
lg = load("lgd_model")
sc = load("scorecard_model")
pd_model, FEATS, CATEGORICAL = lg_pd["model"], lg_pd["features"], lg_pd["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    test[c] = pd.Categorical(test[c].where(test[c].isin(cats)), categories=cats)

def clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)

def calibrate(p):
    if cal["method"] == "intercept_shift":
        return expit(logit(clip(p)) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(clip(p)) + cal["platt_intercept"])

def predict_lgd(X):
    if lg["choice"] == "baseline":
        return np.full(len(X), lg["baseline"])
    if lg["choice"] == "two_stage":
        pf = lg["stage1"].predict_proba(X)[:, 1]
        return pf + (1 - pf) * lg["stage2"].predict(X)
    return lg["lgbm"].predict(X)

print(f"Test portfolio: {len(test):,} loans, ${test['funded_amnt'].sum() / 1e9:.2f}bn lent")

# =====================================================================
# Step 2: final scores on 2015
# =====================================================================
y = test["default"].values
raw = pd_model.predict_proba(test[FEATS])[:, 1]
pd_cal = calibrate(raw)

preds = {
    "Scorecard (WoE + LR)": sc["model"].predict_proba(test_w[sc["features"]])[:, 1],
    "LightGBM": raw,
    "Lending Club int_rate": test["int_rate"].values,
}
print()
for name, p in preds.items():
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    print(f"{name:22} AUC={auc:.3f}  Gini={2*auc - 1:.3f}  KS={(tpr - fpr).max():.3f}")

rng = np.random.default_rng(42)
lc_rate = test["int_rate"].values
diffs = []
for _ in range(200):
    i = rng.integers(0, len(y), len(y))
    diffs.append(roc_auc_score(y[i], raw[i]) - roc_auc_score(y[i], lc_rate[i]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nLightGBM minus Lending Club AUC: {np.mean(diffs):+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})")
print(f"Mean calibrated PD: {pd_cal.mean():.2%}   Actual 2015 default rate: {y.mean():.2%}   "
      f"Ratio: {y.mean() / pd_cal.mean():.2f}")

# =====================================================================
# Step 3: timing of default for every loan (Cox shape x LightGBM level)
# =====================================================================
def prep_cox(d):
    X = d[surv["features"]].astype(float).copy()
    X["annual_inc"] = np.log1p(X["annual_inc"])
    X = X.fillna(surv["stats"]["median"])
    return (X - surv["stats"]["mean"]) / surv["stats"]["std"]

times = np.arange(1, 37)
S = surv["cox"].predict_survival_function(prep_cox(test), times=times).T.values
marg_cox = -np.diff(np.hstack([np.ones((len(S), 1)), S]), axis=1)
shape = marg_cox / marg_cox.sum(axis=1, keepdims=True)

print(f"\nShare of lifetime default risk in the first 12 months: {shape[:, :12].sum(1).mean():.1%}")

# =====================================================================
# Step 4: EAD, LGD and discounting month by month
# =====================================================================
r = (test["int_rate"].values / 1200)[:, None]
t = times[None, :]
balance_start = ((1 + r) ** 36 - (1 + r) ** (t - 1)) / ((1 + r) ** 36 - 1)
EAD_t = balance_start * test["funded_amnt"].values[:, None]
DF = (1 + r) ** (-t)
LGD = np.clip(predict_lgd(test[FEATS]), 0, 1)

def ecl(pd_vec, lgd_vec, months=36, discount=True):
    m = pd_vec[:, None] * shape
    loss = m * lgd_vec[:, None] * EAD_t
    if discount:
        loss = loss * DF
    return loss[:, :months].sum(axis=1)

test["ECL_12m"] = ecl(pd_cal, LGD, months=12)
test["ECL_life"] = ecl(pd_cal, LGD)

# =====================================================================
# Step 5: portfolio totals and the check against realised losses
# =====================================================================
exposure = test["funded_amnt"].sum()
print(f"\n12-month ECL:  ${test['ECL_12m'].sum() / 1e6:,.1f}m  "
      f"({test['ECL_12m'].sum() / exposure:.2%} of lending)")
print(f"Lifetime ECL:  ${test['ECL_life'].sum() / 1e6:,.1f}m  "
      f"({test['ECL_life'].sum() / exposure:.2%} of lending)")

ead_v = (test["funded_amnt"] - test["total_rec_prncp"]).clip(lower=0)
net_v = (test["recoveries"] - test["collection_recovery_fee"]).clip(lower=0)
actual = np.where(y == 1, (ead_v - net_v).clip(lower=0), 0.0)
undisc = ecl(pd_cal, LGD, discount=False)

print(f"\nUndiscounted lifetime ECL: ${undisc.sum() / 1e6:,.1f}m")
print(f"Actual realised loss:      ${actual.sum() / 1e6:,.1f}m")
print(f"Ratio actual / predicted:  {actual.sum() / undisc.sum():.2f}")

pd_decile = pd.qcut(pd.Series(pd_cal).rank(method="first"), 10, labels=False).values
dec = (pd.DataFrame({"decile": pd_decile, "predicted": undisc, "actual": actual})
         .groupby("decile").sum().div(1e6).round(2))
print("\nPredicted vs actual loss by PD decile ($m):")
print(dec)

# =====================================================================
# Step 6: stress test with probability-weighted scenarios
# =====================================================================
def stress_pd(p, factor):
    target = min(p.mean() * factor, 0.95)
    s = brentq(lambda d: expit(logit(clip(p)) + d).mean() - target, -5, 5)
    return expit(logit(clip(p)) + s)

lgd_mean = LGD.mean()
downturn_factor = max(1.0, lg["downturn"] / lgd_mean)

scenarios = {
    # name: (PD multiplier, LGD multiplier, probability weight)
    "Base":    (1.0, 1.0, 0.60),
    "Adverse": (1.5, 1 + (downturn_factor - 1) / 2, 0.30),
    "Severe":  (2.0, downturn_factor, 0.10),
}

results = {}
print()
for name, (pd_mult, lgd_mult, w) in scenarios.items():
    pd_s = stress_pd(pd_cal, pd_mult)
    lgd_s = np.clip(LGD * lgd_mult, 0, 1)
    results[name] = {"pd": pd_s, "lgd": lgd_s, "weight": w,
                     "ecl_12m": ecl(pd_s, lgd_s, months=12),
                     "ecl_life": ecl(pd_s, lgd_s)}
    print(f"{name:8} mean PD={pd_s.mean():6.2%}  "
          f"lifetime ECL=${results[name]['ecl_life'].sum() / 1e6:8.1f}m  weight={w:.0%}")

weighted = sum(res["weight"] * res["ecl_life"].sum() for res in results.values())
print(f"\nProbability-weighted lifetime ECL: ${weighted / 1e6:,.1f}m  "
      f"(vs base ${results['Base']['ecl_life'].sum() / 1e6:,.1f}m)")

# =====================================================================
# Step 7: IFRS 9 staging under the Severe scenario
# =====================================================================
severe_pd = results["Severe"]["pd"]
sicr = (severe_pd >= 2 * pd_cal) & (severe_pd - pd_cal >= 0.05)
print(f"\nLoans moving to Stage 2 under Severe: {sicr.mean():.1%}")

sev = results["Severe"]
all_stage1 = sev["ecl_12m"].sum()
staged = np.where(sicr, sev["ecl_life"], sev["ecl_12m"]).sum()
print(f"Severe reserve if everything stayed Stage 1: ${all_stage1 / 1e6:,.1f}m")
print(f"Severe reserve with staging:                 ${staged / 1e6:,.1f}m  "
      f"(+{staged / all_stage1 - 1:.0%})")

# =====================================================================
# Step 8: chart, summary table, per-loan results
# =====================================================================
names = list(results)
plt.figure(figsize=(6, 4))
plt.bar(names, [results[n]["ecl_life"].sum() / 1e6 for n in names])
plt.axhline(weighted / 1e6, linestyle="--", color="grey", label="Probability-weighted")
plt.ylabel("Lifetime ECL ($m)")
plt.title("Lifetime ECL by scenario – 2015 portfolio")
plt.legend()
plt.savefig("reports/figures/ecl_scenarios.png", dpi=120, bbox_inches="tight")
plt.close()

summary = pd.DataFrame({
    "metric": ["Loans", "Exposure ($m)", "12m ECL ($m)", "Lifetime ECL ($m)",
               "Weighted lifetime ECL ($m)", "Actual loss ($m)", "Stage 2 share (Severe)"],
    "value": [len(test), exposure / 1e6, test["ECL_12m"].sum() / 1e6,
              test["ECL_life"].sum() / 1e6, weighted / 1e6, actual.sum() / 1e6, sicr.mean()],
})
summary.to_csv("reports/ecl_summary.csv", index=False)
test[["ECL_12m", "ECL_life"]].assign(PD=pd_cal, LGD=LGD).to_parquet("data/processed/test_ecl.parquet")
print("\nSaved ECL summary, per-loan results, and scenario chart.")