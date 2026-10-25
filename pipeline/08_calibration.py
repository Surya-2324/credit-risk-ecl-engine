import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")

with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

cal = valid[valid["issue_d"] < "2014-07-01"]
ev  = valid[valid["issue_d"] >= "2014-07-01"]

y_cal, y_ev = cal["default"].values, ev["default"].values
p_cal = model.predict_proba(cal[FEATS])[:, 1]
p_ev  = model.predict_proba(ev[FEATS])[:, 1]

print(f"Calibration half: {len(cal):,} loans, default rate {y_cal.mean():.2%}")
print(f"Evaluation half:  {len(ev):,} loans, default rate {y_ev.mean():.2%}")

def report(name, y, p):
    df = pd.DataFrame({"p": p, "y": y})
    df["dec"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
    g = df.groupby("dec").agg(pred=("p", "mean"), act=("y", "mean"), n=("y", "size"))
    z = (g["act"] - g["pred"]) / np.sqrt(g["pred"] * (1 - g["pred"]) / g["n"])
    reds = int((z.abs() >= 2.58).sum())
    ece = (g["n"] * (g["act"] - g["pred"]).abs()).sum() / g["n"].sum()
    print(f"{name:26} AUC={roc_auc_score(y, p):.3f}  Brier={brier_score_loss(y, p):.4f}  "
          f"mean PD={p.mean():.2%} (actual {y.mean():.2%})  ECE={ece:.4f}  RED={reds}/10")

def clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)

# 1. Intercept shift: find the one number that makes the average PD match reality
def shift(p, d):
    return expit(logit(clip(p)) + d)

delta = brentq(lambda d: shift(p_cal, d).mean() - y_cal.mean(), -3, 3)
print(f"Intercept shift: {delta:+.3f} on the log-odds scale")

# 2. Platt scaling: logistic regression on the model's log-odds
platt = LogisticRegression(C=1e6)
platt.fit(logit(clip(p_cal)).reshape(-1, 1), y_cal)
print(f"Platt: slope={platt.coef_[0][0]:.3f}, intercept={platt.intercept_[0]:+.3f}")

# 3. Isotonic regression: flexible, never reverses the ranking
iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
iso.fit(p_cal, y_cal)

params = model.get_params()
params.update(n_estimators=model.best_iteration_, class_weight="balanced")
balanced = lgb.LGBMClassifier(**params)
balanced.fit(train[FEATS], train["default"])

pb_cal = balanced.predict_proba(cal[FEATS])[:, 1]
pb_ev  = balanced.predict_proba(ev[FEATS])[:, 1]

platt_b = LogisticRegression(C=1e6).fit(logit(clip(pb_cal)).reshape(-1, 1), y_cal)

print()
report("Balanced (SMOTE-style)", y_ev, pb_ev)
report("Balanced + Platt", y_ev, platt_b.predict_proba(logit(clip(pb_ev)).reshape(-1, 1))[:, 1])

with open("data/processed/calibrator.pkl", "wb") as f:
    pickle.dump({"method": "platt",
                 "platt_slope": platt.coef_[0][0],
                 "platt_intercept": platt.intercept_[0],
                 "intercept_shift": delta}, f)
print("\nSaved calibrator.")

# ---------- Step 5: compare the three methods on Jul–Dec 2014 ----------
print()
report("Raw LightGBM", y_ev, p_ev)
report("+ Intercept shift", y_ev, shift(p_ev, delta))
report("+ Platt scaling", y_ev, platt.predict_proba(logit(clip(p_ev)).reshape(-1, 1))[:, 1])
report("+ Isotonic", y_ev, iso.predict(p_ev))

# ---------- Step 7: final calibrator, learned on ALL of 2014 ----------
y_all = valid["default"].values
p_all = model.predict_proba(valid[FEATS])[:, 1]

delta_all = brentq(lambda d: shift(p_all, d).mean() - y_all.mean(), -3, 3)
platt_all = LogisticRegression(C=1e6).fit(logit(clip(p_all)).reshape(-1, 1), y_all)

METHOD = "platt"   # change after reading Step 5 (see rule below)

with open("data/processed/calibrator.pkl", "wb") as f:
    pickle.dump({"method": METHOD,
                 "intercept_shift": delta_all,
                 "platt_slope": platt_all.coef_[0][0],
                 "platt_intercept": platt_all.intercept_[0]}, f)
print(f"\nSaved final calibrator ({METHOD}), learned on all of 2014: "
      f"shift={delta_all:+.3f}, Platt slope={platt_all.coef_[0][0]:.3f}")