import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

train_w = pd.read_parquet("data/processed/train_woe.parquet")
valid_w = pd.read_parquet("data/processed/valid_woe.parquet")
with open("data/processed/scorecard_model.pkl", "rb") as f:
    saved = pickle.load(f)
model,features = saved["model"],saved["features"]
y=valid_w["default"].values
p = model.predict_proba(valid_w[features])[:,1]

os.makedirs("reports/figures",exist_ok=True)

auc = roc_auc_score(y,p)
rng = np.random.default_rng(42)
boot=[]
for _ in range(200):
    idx = rng.integers(0,len(y),len(y))
    boot.append(roc_auc_score(y[idx],p[idx]))
lo,hi =np.percentile(boot,[2.5,97.5])

print(f"AUC  = {auc:.3f}  (95% CI {lo:.3f} to {hi:.3f})")
print(f"Gini = {2*auc - 1:.3f}  (95% CI {2*lo - 1:.3f} to {2*hi - 1:.3f})")

fpr, tpr, _ = roc_curve(y, p)

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"Scorecard (AUC {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="grey", label="Random guessing")
plt.xlabel("Share of good borrowers wrongly flagged")
plt.ylabel("Share of defaulters caught")
plt.title("ROC curve – validation 2014")
plt.legend()
plt.savefig("reports/figures/roc.png", dpi=120, bbox_inches="tight")
plt.close()

order = np.argsort(-p)
cum_bad = np.cumsum(y[order])/y.sum()
cum_good = np.cumsum(1-y[order])/(1-y).sum()
share = np.arange(1,len(y)+1)/len(y)

gap  = cum_bad-cum_good
k = np.argmax(gap)
print(f"KS = {gap[k]:.3f}, reached when declining the riskiest {share[k]:.1%}")

for pct in [0.10,0.20,0.30]:
    i = int(pct*len(y)) - 1
    print(f"Decline riskiest {pct:.0%}: catch {cum_bad[i]:.1%} of defaulters, "
          f"lose {cum_good[i]:.1%} of good borrowers")
plt.figure(figsize=(6, 4))
plt.plot(share, cum_bad, label="Defaulters caught")
plt.plot(share, cum_good, label="Good borrowers lost")
plt.axvline(share[k], linestyle="--", color="grey", label=f"KS = {gap[k]:.3f}")
plt.xlabel("Share of applicants declined (riskiest first)")
plt.ylabel("Cumulative share")
plt.title("KS chart – validation 2014")
plt.legend()
plt.savefig("reports/figures/ks.png", dpi=120, bbox_inches="tight")
plt.close()

y_train_rate = train_w["default"].mean()
brier = brier_score_loss(y, p)
brier_ref = brier_score_loss(y, np.full(len(y), y_train_rate))

print(f"Mean predicted PD: {p.mean():.2%}   Actual: {y.mean():.2%}   "
      f"Ratio: {y.mean() / p.mean():.2f}")
print(f"Brier: {brier:.4f}   Naive Brier: {brier_ref:.4f}   "
      f"Brier skill score: {1 - brier / brier_ref:.3f}")

import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")

# 1. Benchmark: Lending Club's own pricing
print("Benchmark AUC (int_rate):", round(roc_auc_score(valid["default"], valid["int_rate"]), 3))

# 2. revol_util
print("\nrevol_util dtype:", train["revol_util"].dtype)
print("revol_util missing share:", round(train["revol_util"].isna().mean(), 3))
print(train["revol_util"].describe())

# 3. Zero-heavy features
for c in ["pub_rec", "delinq_2yrs"]:
    print(f"\n{c} value counts:")
    print(train[c].value_counts().head(5))

# 4. credit_history_years
print("\ncredit_history_years missing share:", round(train["credit_history_years"].isna().mean(), 3))
print(train["credit_history_years"].describe())
    