import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

os.makedirs("reports/figures", exist_ok=True)

# ---------- Step 1: load the clean splits ----------
train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
test  = pd.read_parquet("data/processed/test.parquet")

# ---------- Step 2: features and required directions ----------
# +1: higher value -> higher PD,  -1: higher value -> lower PD,  0: free
NUMERIC = {
    "fico": -1,
    "dti": +1,
    "annual_inc": -1,
    "loan_to_income": +1,
    "credit_history_years": -1,
    "inq_last_6mths": +1,
    "revol_util": +1,
    "delinq_2yrs": +1,
    "pub_rec": +1,
    "pub_rec_bankruptcies": +1,
    "mort_acc": -1,
    "emp_years": 0,
    "revol_bal": 0,
    "open_acc": 0,
    "total_acc": 0,
    "loan_amnt": 0,
}
CATEGORICAL = ["home_ownership", "purpose", "verification_status", "addr_state"]

FEATS = list(NUMERIC) + CATEGORICAL
constraints = list(NUMERIC.values()) + [0] * len(CATEGORICAL)

# ---------- Step 3: consistent categories (unseen -> missing) ----------
for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid, test):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

# ---------- Step 4: constrained LightGBM ----------
params = dict(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1,
)

model = lgb.LGBMClassifier(monotone_constraints=constraints, **params)
model.fit(
    train[FEATS], train["default"],
    eval_set=[(valid[FEATS], valid["default"])],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
)
print("Best number of trees (constrained):", model.best_iteration_)

# ---------- Step 5: unconstrained twin ----------
free = lgb.LGBMClassifier(**params)
free.fit(
    train[FEATS], train["default"],
    eval_set=[(valid[FEATS], valid["default"])],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
print("Best number of trees (free):", free.best_iteration_)

# ---------- Step 6: compare all models on 2014 ----------
with open("data/processed/scorecard_model.pkl", "rb") as f:
    sc = pickle.load(f)
valid_w = pd.read_parquet("data/processed/valid_woe.parquet")

y = valid["default"].values
preds = {
    "Scorecard (WoE + LR)": sc["model"].predict_proba(valid_w[sc["features"]])[:, 1],
    "LightGBM constrained": model.predict_proba(valid[FEATS])[:, 1],
    "LightGBM free": free.predict_proba(valid[FEATS])[:, 1],
    "Lending Club int_rate": valid["int_rate"].values,
}

print()
for name, p in preds.items():
    auc_val = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    line = f"{name:22} AUC={auc_val:.3f}  Gini={2*auc_val-1:.3f}  KS={(tpr-fpr).max():.3f}"
    if name != "Lending Club int_rate":
        line += f"  mean PD={p.mean():.2%} (actual {y.mean():.2%})"
    print(line)

# ---------- Step 7: prove the FICO constraint holds ----------
def average_curve(m, data, col, grid, n=3000):
    sample = data.sample(n, random_state=42).copy()
    out = []
    for v in grid:
        sample[col] = v
        out.append(m.predict_proba(sample[FEATS])[:, 1].mean())
    return np.array(out)

grid = np.quantile(train["fico"].dropna(), np.linspace(0.02, 0.98, 25))
curve_c = average_curve(model, valid, "fico", grid)
curve_f = average_curve(free, valid, "fico", grid)

print("\nConstrained never rises with FICO:", bool(np.all(np.diff(curve_c) <= 1e-9)))
print("Free model never rises with FICO: ", bool(np.all(np.diff(curve_f) <= 1e-9)))

plt.figure(figsize=(6, 4))
plt.plot(grid, curve_c, label="Constrained")
plt.plot(grid, curve_f, label="Free", alpha=0.7)
plt.xlabel("FICO")
plt.ylabel("Average predicted PD")
plt.title("Effect of FICO on predicted PD")
plt.legend()
plt.savefig("reports/figures/fico_monotonic.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------- Step 8: what the model relies on ----------
imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATS)
print("\nTop 10 features (share of total gain):")
print((imp / imp.sum()).sort_values(ascending=False).round(3).head(10))

# ---------- Step 9: save ----------
with open("data/processed/lgbm_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATS,
                 "categorical": CATEGORICAL, "constraints": NUMERIC}, f)
print("\nSaved LightGBM model.")

feats_no_state = [f for f in FEATS if f != "addr_state"]
cons_no_state = [c for f, c in zip(FEATS, constraints) if f != "addr_state"]

m_ns = lgb.LGBMClassifier(monotone_constraints=cons_no_state, **params)
m_ns.fit(
    train[feats_no_state], train["default"],
    eval_set=[(valid[feats_no_state], valid["default"])],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
p_ns = m_ns.predict_proba(valid[feats_no_state])[:, 1]
print(f"Without addr_state: AUC={roc_auc_score(y, p_ns):.3f}")