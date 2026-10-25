import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

rng = np.random.default_rng(42)
n = 10_000

df = pd.DataFrame({
    "credit_score": rng.normal(680, 60, n).clip(300, 850),
    "dti": rng.uniform(0, 45, n),
    "income": rng.lognormal(10.8, 0.5, n),
    "loan_amount": rng.uniform(1_000, 35_000, n),
})

logit = (-3.0
         - 0.02 * (df["credit_score"] - 680)
         + 0.05 * (df["dti"] - 20)
         - 0.5 * np.log(df["income"] / 50_000))

df["true_pd"] = 1 / (1 + np.exp(-logit))
df["default"] = rng.binomial(1, df["true_pd"])
print(f"Default rate: {df['default'].mean():.2%}")

# Accuracy 
y = df['default']
nobady_defaults = np.zeros(n)
print(f'Accuracy:{accuracy_score(y, nobady_defaults):.2%}')
print(f"AUC: {roc_auc_score(y, nobady_defaults):.2%}")

# Expected Loss for portfolio
LGD =0.45
df['EAD'] =df['loan_amount']
df ['EL'] = df['true_pd'] * LGD * df['EAD'] # EL =pd x ead x lgd
print(f"Portfolio EL: £{df['EL'].sum():,.0f}")
print(f"EL as % of lending: {df['EL'].sum() / df['EAD'].sum():.2%}")

#  Same AUC but different money

model_a = df["true_pd"]           # perfectly calibrated probabilities
model_b = df["true_pd"] ** 0.5    # same ranking, but inflated probabilities

for name, p in [("A (calibrated)", model_a), ("B (inflated)", model_b)]:
    el = (p * LGD * df["EAD"]).sum()
    print(f"{name:15} AUC={roc_auc_score(y, p):.3f}  "
          f"Brier={brier_score_loss(y, p):.4f}  "
          f"Mean PD={p.mean():.2%}  EL=£{el:,.0f}")
#  calibration table 
df["pd_b"] = model_b
df["bucket"] = pd.qcut(df["pd_b"], 10, labels=False)

calib = df.groupby("bucket").agg(predicted=("pd_b", "mean"),
                                 actual=("default", "mean"))
print(calib.round(3))