import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
test = pd.read_parquet("data/processed/test.parquet")

with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
with open("data/processed/calibrator.pkl", "rb") as f:
    cal = pickle.load(f)
model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid, test):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

def calibrate(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if cal["method"] == "intercept_shift":
        return expit(logit(p) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(p) + cal["platt_intercept"])

for d in (train, valid, test):
    d["pd"] = calibrate(model.predict_proba(d[FEATS])[:, 1])

prod = pd.concat([valid, test], ignore_index=True)
prod["quarter"] = prod["issue_d"].dt.to_period("Q")
print(prod["quarter"].value_counts().sort_index())

def psi(expected, actual, bins=10, eps=1e-4):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return psi_cat(expected, actual, eps)
    edges[0], edges[-1] = -np.inf, np.inf
    e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
    a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=eps), a.clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())

def psi_cat(expected, actual, eps=1e-4):
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e = e.reindex(idx, fill_value=0).clip(lower=eps)
    a = a.reindex(idx, fill_value=0).clip(lower=eps)
    return float(((a - e) * np.log(a / e)).sum())

csi = {}
for f in FEATS:
    fn = psi_cat if f in CATEGORICAL else psi
    csi[f] = fn(train[f], test[f])
csi = pd.Series(csi).sort_values(ascending=False)

def flag(v):
    return "RED" if v > 0.25 else "AMBER" if v > 0.10 else "GREEN"

print("\nFeature stability, train vs 2015:")
print(pd.DataFrame({"CSI": csi.round(3), "status": csi.map(flag)}))

miss_train = train[FEATS].isna().mean()
miss_q = prod.groupby("quarter")[FEATS].apply(lambda g: g.isna().mean())
jump = (miss_q - miss_train).abs().max()
print("\nLargest change in missing share vs training (top 5):")
print(jump.sort_values(ascending=False).head(5).round(3))


rows = []
for q, g in prod.groupby("quarter"):
    rows.append({
        "quarter": str(q),
        "loans": len(g),
        "score_psi": psi(train["pd"], g["pd"]),
        "mean_pd": g["pd"].mean(),
        "actual_dr": g["default"].mean(),
        "auc": roc_auc_score(g["default"], g["pd"]),
    })
report = pd.DataFrame(rows)
report["calib_ratio"] = report["actual_dr"] / report["mean_pd"]

def status(row):
    if row["score_psi"] > 0.25 or not 0.90 <= row["calib_ratio"] <= 1.10:
        return "RED"
    if row["score_psi"] > 0.10 or not 0.95 <= row["calib_ratio"] <= 1.05:
        return "AMBER"
    return "GREEN"

report["status"] = report.apply(status, axis=1)
print("\nQuarterly monitoring report:")
print(report.round(3).to_string(index=False))
report.to_csv("reports/monitoring_report.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(report["quarter"], report["score_psi"], marker="o")
axes[0].axhline(0.10, color="orange", linestyle="--", label="Investigate (0.10)")
axes[0].axhline(0.25, color="red", linestyle="--", label="Act (0.25)")
axes[0].set_title("Score PSI vs training")
axes[0].legend()

axes[1].plot(report["quarter"], report["calib_ratio"], marker="o")
axes[1].axhline(1.0, color="grey")
axes[1].axhspan(0.95, 1.05, color="green", alpha=0.1, label="Green zone")
axes[1].set_title("Calibration ratio (actual / predicted)")
axes[1].legend()

for ax in axes:
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("reports/figures/monitoring.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved monitoring report and chart.")