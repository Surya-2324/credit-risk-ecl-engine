import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

X = valid[FEATS].sample(20_000, random_state=42)
contrib = model.predict(X, pred_contrib=True)
shap_vals = pd.DataFrame(contrib[:, :-1], columns=FEATS, index=X.index)
base = contrib[0, -1]
print(f"Base value (average log-odds): {base:.3f}")

raw_pd = model.predict_proba(X)[:, 1]
rebuilt = base + shap_vals.sum(axis=1)
print(f"Largest gap between model and SHAP sum: {np.abs(logit(raw_pd) - rebuilt).max():.6f}")

shap_imp = shap_vals.abs().mean()
gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATS)

compare = pd.DataFrame({
    "SHAP share": shap_imp / shap_imp.sum(),
    "Gain share": gain / gain.sum(),
}).sort_values("SHAP share", ascending=False)
print(compare.round(3))

plt.figure(figsize=(6, 5))
compare["SHAP share"].iloc[::-1].plot(kind="barh")
plt.xlabel("Share of mean |SHAP|")
plt.title("What drives predicted risk – 2014 sample")
plt.savefig("reports/figures/shap_importance.png", dpi=120, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["fico", "dti", "revol_util"]):
    ax.scatter(X[col], shap_vals[col], s=2, alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel(col)
    ax.set_ylabel("SHAP (log-odds)")
    ax.set_title(f"Effect of {col}")
plt.tight_layout()
plt.savefig("reports/figures/shap_dependence.png", dpi=120, bbox_inches="tight")
plt.close()

REASONS = {
    "fico": "Credit score of {v:.0f}",
    "dti": "Debt-to-income ratio of {v:.1f}%",
    "annual_inc": "Annual income of ${v:,.0f}",
    "loan_to_income": "Loan amount is {v:.0%} of annual income",
    "credit_history_years": "Credit history of {v:.1f} years",
    "inq_last_6mths": "{v:.0f} credit inquiries in the last 6 months",
    "revol_util": "Credit card utilisation of {v:.0f}%",
    "delinq_2yrs": "{v:.0f} delinquencies in the last 2 years",
    "pub_rec": "{v:.0f} public records",
    "pub_rec_bankruptcies": "{v:.0f} bankruptcies on record",
    "mort_acc": "{v:.0f} mortgage accounts",
    "emp_years": "Employment length of {v:.0f} years",
    "revol_bal": "Revolving balance of ${v:,.0f}",
    "open_acc": "{v:.0f} open credit lines",
    "total_acc": "{v:.0f} total credit lines",
    "loan_amnt": "Loan amount of ${v:,.0f}",
    "home_ownership": "Home ownership: {v}",
    "purpose": "Loan purpose: {v}",
    "verification_status": "Income verification: {v}",
    "addr_state": "State of residence: {v}",
}

def describe(idx, f):
    v = X.at[idx, f]
    return f"{f}: not provided" if pd.isna(v) else REASONS[f].format(v=v)

def reason_codes(idx, k=4):
    s = shap_vals.loc[idx].sort_values(ascending=False)
    return [f"{describe(idx, f)}  (+{val:.2f})" for f, val in s.items() if val > 0][:k]

def strengths(idx, k=3):
    s = shap_vals.loc[idx].sort_values()
    return [f"{describe(idx, f)}  ({val:.2f})" for f, val in s.items() if val < 0][:k]
p = model.predict_proba(X)[:, 1]
order = np.argsort(p)
examples = {
    "Riskiest": X.index[order[-1]],
    "Median": X.index[order[len(order) // 2]],
    "Safest": X.index[order[0]],
}

for label, idx in examples.items():
    print(f"\n=== {label} applicant: predicted PD {p[X.index.get_loc(idx)]:.1%} ===")
    print("Reasons risk is higher:")
    for r in reason_codes(idx) or ["none"]:
        print("  -", r)
    print("Reasons risk is lower:")
    for r in strengths(idx) or ["none"]:
        print("  -", r)

idx = examples["Riskiest"]
s = shap_vals.loc[idx]
top = s.reindex(s.abs().sort_values(ascending=False).index[:8]).iloc[::-1]

plt.figure(figsize=(7, 4))
colors = ["#c0392b" if val > 0 else "#2e86c1" for val in top]
plt.barh([describe(idx, f) for f in top.index], top.values, color=colors)
plt.axvline(0, color="grey", linewidth=0.8)
plt.xlabel("Push on log-odds of default (red = riskier, blue = safer)")
plt.title("Why this applicant is high risk")
plt.savefig("reports/figures/shap_riskiest.png", dpi=120, bbox_inches="tight")
plt.close()

shap_vals.to_parquet("data/processed/shap_valid_sample.parquet")
with open("data/processed/reason_templates.pkl", "wb") as f:
    pickle.dump(REASONS, f)
print("\nSaved SHAP values and reason templates.")