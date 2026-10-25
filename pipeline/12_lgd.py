import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
FEATS, CATEGORICAL = saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

def add_lgd(d):
    d = d[d["default"] == 1].copy()
    d["ead"] = (d["funded_amnt"] - d["total_rec_prncp"]).clip(lower=0)
    d = d[d["ead"] > 0].copy()
    net = (d["recoveries"] - d["collection_recovery_fee"]).clip(lower=0)
    d["lgd"] = ((d["ead"] - net) / d["ead"]).clip(0, 1)
    return d

tr, va = add_lgd(train), add_lgd(valid)
print(f"Train defaulters: {len(tr):,}   Valid defaulters: {len(va):,}")

print(tr["lgd"].describe().round(3))
print(f"Share with no recovery at all (LGD = 100%): {(tr['lgd'] >= 0.999).mean():.1%}")
w = (tr["lgd"] * tr["ead"]).sum() / tr["ead"].sum()
print(f"Exposure-weighted LGD: {w:.1%}")

plt.figure(figsize=(6, 4))
plt.hist(tr["lgd"], bins=50)
plt.xlabel("LGD")
plt.ylabel("Number of defaulted loans")
plt.title("Distribution of LGD – train defaulters")
plt.savefig("reports/figures/lgd_hist.png", dpi=120, bbox_inches="tight")
plt.close()

both = pd.concat([tr, va])
yearly = both.groupby("year").apply(
    lambda g: pd.Series({"defaults": len(g),
                         "mean_lgd": g["lgd"].mean(),
                         "weighted_lgd": (g["lgd"] * g["ead"]).sum() / g["ead"].sum()}),
    include_groups=False)
print(yearly.round(3))

baseline = tr["lgd"].mean()
p_base = np.full(len(va), baseline)

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

lgd_model = lgb.LGBMRegressor(**params)
lgd_model.fit(
    tr[FEATS], tr["lgd"],
    eval_set=[(va[FEATS], va["lgd"])],
    eval_metric="l2",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
)
p_lgbm = lgd_model.predict(va[FEATS])
print("Best number of trees:", lgd_model.best_iteration_)

tr["full_loss"] = (tr["lgd"] >= 0.999).astype(int)
stage_params = {k: v for k, v in params.items() if k != "objective"}
stage_params["n_estimators"] = 300

stage1 = lgb.LGBMClassifier(**stage_params)
stage1.fit(tr[FEATS], tr["full_loss"])

partial = tr[tr["full_loss"] == 0]
stage2 = lgb.LGBMRegressor(objective="cross_entropy", **stage_params)
stage2.fit(partial[FEATS], partial["lgd"])

p_full = stage1.predict_proba(va[FEATS])[:, 1]
p_two = p_full * 1.0 + (1 - p_full) * stage2.predict(va[FEATS])

def evaluate(name, y, p, ead):
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    r2 = r2_score(y, p)
    print(f"{name:12} MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  "
          f"predicted loss=${(p * ead).sum() / 1e6:.1f}m  actual=${(y * ead).sum() / 1e6:.1f}m")

y_va, ead_va = va["lgd"].values, va["ead"].values
print()
evaluate("Baseline", y_va, p_base, ead_va)
evaluate("LightGBM", y_va, p_lgbm, ead_va)
evaluate("Two-stage", y_va, p_two, ead_va)

downturn = yearly.loc[yearly.index <= 2013, "weighted_lgd"].max()
print(f"\nDownturn LGD (worst mature vintage): {downturn:.1%}")

contrib = lgd_model.predict(va[FEATS], pred_contrib=True)
imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=FEATS)
print("\nTop LGD drivers (mean |SHAP|):")
print((imp / imp.sum()).sort_values(ascending=False).round(3).head(8))

va["pred"] = p_lgbm
va["dec"] = pd.qcut(va["pred"].rank(method="first"), 5, labels=False)
print("\nPredicted vs actual LGD by quintile:")
print(va.groupby("dec").agg(predicted=("pred", "mean"), actual=("lgd", "mean"), n=("lgd", "size")).round(3))

best_name = "lightgbm"   # change to "two_stage" or "baseline" based on Step 7
with open("data/processed/lgd_model.pkl", "wb") as f:
    pickle.dump({"choice": best_name, "lgbm": lgd_model,
                 "stage1": stage1, "stage2": stage2,
                 "baseline": baseline, "downturn": downturn,
                 "features": FEATS}, f)
print("\nSaved LGD model.")