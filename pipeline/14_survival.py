import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")

def add_survival(d):
    d = d.copy()
    last = pd.to_datetime(d["last_pymnt_d"], format="%b-%Y", errors="coerce")
    months = (last.dt.year - d["issue_d"].dt.year) * 12 + (last.dt.month - d["issue_d"].dt.month)
    months = months.fillna(0)
    d["event"] = d["default"]
    d["duration"] = np.where(d["event"] == 1, months + 1, months).clip(1, 36)
    return d

tr, va = add_survival(train), add_survival(valid)
print(tr[["duration", "event"]].describe().round(2))

kmf = KaplanMeierFitter()
kmf.fit(tr["duration"], tr["event"], label="All loans")

print(f"\nKaplan–Meier PD by month 12: {1 - kmf.predict(12):.2%}")
print(f"Kaplan–Meier PD by month 36: {1 - kmf.predict(36):.2%}")

tr["fico_band"] = pd.qcut(tr["fico"], 4, labels=["Lowest 25%", "Q2", "Q3", "Highest 25%"])
fig, ax = plt.subplots(figsize=(7, 4))
for band, g in tr.groupby("fico_band", observed=True):
    KaplanMeierFitter().fit(g["duration"], g["event"], label=f"FICO {band}") \
        .plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel("Months since issue")
ax.set_ylabel("Share not yet defaulted")
ax.set_title("Survival by FICO band – train")
plt.savefig("reports/figures/km_fico.png", dpi=120, bbox_inches="tight")
plt.close()

et = kmf.event_table
hazard = (et["observed"] / et["at_risk"]).loc[1:36]

plt.figure(figsize=(7, 4))
plt.bar(hazard.index, hazard.values)
plt.xlabel("Month since issue")
plt.ylabel("Monthly default hazard")
plt.title("Risk of defaulting in each month – train")
plt.savefig("reports/figures/hazard.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Highest-risk month: {int(hazard.idxmax())}")

COX_FEATS = ["fico", "dti", "annual_inc", "revol_util", "inq_last_6mths",
             "loan_to_income", "mort_acc", "emp_years", "credit_history_years"]

def prep(d, stats=None):
    X = d[COX_FEATS].astype(float).copy()
    X["annual_inc"] = np.log1p(X["annual_inc"])
    if stats is None:
        median = X.median()
        X = X.fillna(median)
        stats = {"median": median, "mean": X.mean(), "std": X.std()}
    else:
        X = X.fillna(stats["median"])
    return (X - stats["mean"]) / stats["std"], stats

X_tr, stats = prep(tr)
X_va, _ = prep(va, stats)

cox_df = X_tr.assign(duration=tr["duration"].values, event=tr["event"].values)
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col="duration", event_col="event")
print(cph.summary[["exp(coef)", "p"]].round(3))

risk = cph.predict_partial_hazard(X_va)
c_index = concordance_index(va["duration"], -risk, va["event"])
print(f"\nConcordance index on 2014: {c_index:.3f}")

surv = cph.predict_survival_function(X_va, times=[12, 36])
pd12 = 1 - surv.loc[12].values
pd36 = 1 - surv.loc[36].values

actual12 = ((va["event"] == 1) & (va["duration"] <= 12)).mean()
print(f"\n12-month PD: predicted {pd12.mean():.2%}   actual {actual12:.2%}")
print(f"36-month PD: predicted {pd36.mean():.2%}   observed default rate {va['event'].mean():.2%}")
print(f"AUC of lifetime PD for 'ever defaulted': {roc_auc_score(va['event'], pd36):.3f}")

times = np.arange(1, 37)
sample = X_va.sample(5000, random_state=42)
mean_surv = cph.predict_survival_function(sample, times=times).mean(axis=1)
marginal = -np.diff(np.r_[1.0, mean_surv.values])

plt.figure(figsize=(7, 4))
plt.bar(times, marginal)
plt.xlabel("Month")
plt.ylabel("Probability of defaulting in that month")
plt.title("Marginal PD by month – average 2014 borrower")
plt.savefig("reports/figures/marginal_pd.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"Marginal PDs sum to: {marginal.sum():.2%}")

cph.check_assumptions(cox_df.sample(20000, random_state=42), p_value_threshold=0.05, show_plots=False)

with open("data/processed/survival_model.pkl", "wb") as f:
    pickle.dump({"cox": cph, "stats": stats, "features": COX_FEATS,
                 "km_hazard": hazard}, f)
print("\nSaved survival model.")