import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from sklearn.metrics import mean_absolute_error, r2_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
pd_model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

def add_ead(d):
    d = d[d["default"] == 1].copy()
    d["ead"] = (d["funded_amnt"] - d["total_rec_prncp"]).clip(lower=0)
    d["ead_frac"] = (d["ead"] / d["funded_amnt"]).clip(0, 1)

    last = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
    months = (last.dt.year - d["issue_d"].dt.year) * 12 + (last.dt.month - d["issue_d"].dt.month)
    d["months_paid"] = months.fillna(0).clip(0, 36)
    return d

tr, va = add_ead(train), add_ead(valid)
print(f"Train defaulters: {len(tr):,}   Valid defaulters: {len(va):,}")
print(tr[["ead_frac", "months_paid"]].describe().round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(tr["months_paid"], bins=37)
axes[0].set_xlabel("Months of payments before default")
axes[0].set_ylabel("Number of defaults")
axes[0].set_title("When defaults happen")
axes[1].scatter(tr["months_paid"], tr["ead_frac"], s=2, alpha=0.2)
axes[1].set_xlabel("Months of payments before default")
axes[1].set_ylabel("EAD as share of loan")
axes[1].set_title("Later default, less owed")
plt.tight_layout()
plt.savefig("reports/figures/ead_timing.png", dpi=120, bbox_inches="tight")
plt.close()

bands = pd.cut(tr["months_paid"], [-1, 6, 12, 18, 24, 36])
print(tr.groupby(bands, observed=True)["ead_frac"].agg(["mean", "size"]).round(3))

def scheduled_balance(rate_pct, months, n=36):
    r = rate_pct / 1200
    return ((1 + r) ** n - (1 + r) ** months) / ((1 + r) ** n - 1)

tr["sched_frac"] = scheduled_balance(tr["int_rate"], tr["months_paid"])
gap = tr["ead_frac"] - tr["sched_frac"]

print(f"\nCorrelation between actual and scheduled balance: {tr['ead_frac'].corr(tr['sched_frac']):.3f}")
print(f"Average gap (actual − scheduled): {gap.mean():+.3f}")

params = dict(
    objective="cross_entropy",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=300,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

ead_model = lgb.LGBMRegressor(**params)
ead_model.fit(
    tr[FEATS], tr["ead_frac"],
    eval_set=[(va[FEATS], va["ead_frac"])],
    eval_metric="l2",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)

p_base = np.full(len(va), tr["ead_frac"].mean())
p_ead = ead_model.predict(va[FEATS])

for name, p in [("Baseline", p_base), ("LightGBM", p_ead)]:
    print(f"{name:9} MAE={mean_absolute_error(va['ead_frac'], p):.3f}  "
          f"R2={r2_score(va['ead_frac'], p):.3f}  "
          f"predicted EAD=${(p * va['funded_amnt']).sum() / 1e6:.1f}m  "
          f"actual=${va['ead'].sum() / 1e6:.1f}m")


with open("data/processed/calibrator.pkl", "rb") as f:
    cal = pickle.load(f)
with open("data/processed/lgd_model.pkl", "rb") as f:
    lg = pickle.load(f)

def calibrate(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if cal["method"] == "intercept_shift":
        return expit(logit(p) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(p) + cal["platt_intercept"])

def predict_lgd(X):
    if lg["choice"] == "baseline":
        return np.full(len(X), lg["baseline"])
    if lg["choice"] == "two_stage":
        pf = lg["stage1"].predict_proba(X)[:, 1]
        return pf + (1 - pf) * lg["stage2"].predict(X)
    return lg["lgbm"].predict(X)

v = valid.copy()
v["PD"] = calibrate(pd_model.predict_proba(v[FEATS])[:, 1])
v["LGD"] = predict_lgd(v[FEATS])
v["EAD"] = ead_model.predict(v[FEATS]) * v["funded_amnt"]
v["ECL"] = v["PD"] * v["LGD"] * v["EAD"]

ead_v = (v["funded_amnt"] - v["total_rec_prncp"]).clip(lower=0)
net_v = (v["recoveries"] - v["collection_recovery_fee"]).clip(lower=0)
v["actual_loss"] = np.where(v["default"] == 1, (ead_v - net_v).clip(lower=0), 0.0)

print(f"\nPredicted expected loss: ${v['ECL'].sum() / 1e6:,.1f}m")
print(f"Actual realised loss:    ${v['actual_loss'].sum() / 1e6:,.1f}m")
print(f"Ratio actual / predicted: {v['actual_loss'].sum() / v['ECL'].sum():.2f}")

v["pd_decile"] = pd.qcut(v["PD"].rank(method="first"), 10, labels=False)
print(v.groupby("pd_decile")[["ECL", "actual_loss"]].sum().div(1e6).round(2))

with open("data/processed/ead_model.pkl", "wb") as f:
    pickle.dump({"model": ead_model, "baseline": tr["ead_frac"].mean(),
                 "features": FEATS}, f)
print("\nSaved EAD model.")