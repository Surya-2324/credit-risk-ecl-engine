import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from sklearn.metrics import f1_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")

with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

with open("data/processed/calibrator.pkl", "rb") as f:
    cal = pickle.load(f)

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

def calibrate(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if cal["method"] == "intercept_shift":
        return expit(logit(p) + cal["intercept_shift"])
    return expit(cal["platt_slope"] * logit(p) + cal["platt_intercept"])

bad = train[train["default"] == 1]
ead = (bad["funded_amnt"] - bad["total_rec_prncp"]).clip(lower=0)
net_recovery = (bad["recoveries"] - bad["collection_recovery_fee"]).clip(lower=0)

EADF = (ead / bad["funded_amnt"]).mean()
LGD = (ead - net_recovery).clip(lower=0).sum() / ead.sum()

print(f"Average exposure at default: {EADF:.1%} of the original loan")
print(f"Average LGD: {LGD:.1%} of that exposure is never recovered")

v = valid.copy()
v["pd"] = calibrate(model.predict_proba(v[FEATS])[:, 1])

v["interest"] = v["installment"] * 36 - v["loan_amnt"]
v["loss_if_default"] = LGD * EADF * v["loan_amnt"]
v["exp_profit"] = (1 - v["pd"]) * v["interest"] - v["pd"] * v["loss_if_default"]
v["breakeven_pd"] = v["interest"] / (v["interest"] + v["loss_if_default"])

print(v[["loan_amnt", "int_rate", "pd", "interest", "loss_if_default",
         "exp_profit", "breakeven_pd"]].head().round(3))

ead_v = (v["funded_amnt"] - v["total_rec_prncp"]).clip(lower=0)
loss_v = (ead_v - (v["recoveries"] - v["collection_recovery_fee"]).clip(lower=0)).clip(lower=0)
v["realised"] = np.where(v["default"] == 1, -loss_v, v["interest"])

def summarise(name, approve):
    a = v[approve]
    print(f"{name:34} approve={approve.mean():6.1%}  "
          f"default rate={a['default'].mean():6.2%}  "
          f"profit=£{a['realised'].sum() / 1e6:7.2f}m")

print()
summarise("Approve everyone", np.ones(len(v), dtype=bool))
summarise("Per-loan rule: expected profit > 0", (v["exp_profit"] > 0).values)

order = v["pd"].sort_values().index
cum_profit = v.loc[order, "realised"].cumsum().values
share = np.arange(1, len(v) + 1) / len(v)
best = int(cum_profit.argmax())
pd_cut = v.loc[order, "pd"].iloc[best]
summarise(f"Best single PD cut-off ({pd_cut:.1%})", (v["pd"] <= pd_cut).values)

grid = np.linspace(0.05, 0.50, 46)
f1s = [f1_score(v["default"], (v["pd"] > t).astype(int)) for t in grid]
t_f1 = grid[int(np.argmax(f1s))]
summarise(f"F1-optimal cut-off ({t_f1:.1%})", (v["pd"] <= t_f1).values)
v["rate_band"] = pd.qcut(v["int_rate"], 5)
print(v.groupby("rate_band", observed=True)[["pd", "breakeven_pd"]].mean().round(3))