import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

train_w = pd.read_parquet("data/processed/train_woe.parquet")
valid_w = pd.read_parquet("data/processed/valid_woe.parquet")
with open("data/processed/woe_bins.pkl", "rb") as f:
    saved = pickle.load(f)

features = [c for c in train_w.columns if c != "default" ]
X_tr,y_tr = train_w[features], train_w["default"]
X_va,y_va = valid_w[features], valid_w["default"]
print("Features:",features)


corr = X_tr.corr()
upper = corr.where(np.triu(np.ones(corr.shape,dtype = bool),k=1))
pairs = upper.stack()
print(pairs[pairs.abs()>0.6].round(2))

model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)

coefs = pd.Series(model.coef_[0], index=features).sort_values()
print(coefs.round(3))
print(f"Intercept: {model.intercept_[0]:.3f}")

wrong = coefs[coefs > 0]
print("Wrong-sign features:", list(wrong.index) if len(wrong) else "none")

feature = [f for f in features if f not in wrong.index]
X_tr,X_va = train_w[features], valid_w[features]

def evaluate(y,p,name):
    auc = roc_auc_score(y,p)
    fpr,tpr,_ = roc_curve(y,p)
    ks = (tpr-fpr).max()
    print(f"{name:6} AUC={auc:.3f} Gini = {2*auc -1:.3f} ks = {ks:.3f}"
          f"Brier = {brier_score_loss(y,p):.4f}"
          f"mean PD = {p.mean():.2%} actual={y.mean():.2%}")
p_tr = model.predict_proba(X_tr)[:,1]
p_va = model.predict_proba(X_va)[:,1]
evaluate(y_tr,p_tr,"Train")
evaluate(y_va,p_va,"Valid")


PDO,BASE_SCORE,BASE_ODDS = 20,60,50
FACTOR = PDO/np.log(2)

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
factor = PDO / np.log(2)
offset = BASE_SCORE - factor * np.log(BASE_ODDS)

def to_score(p):
    odds_good = (1 - p) / p
    return offset + factor * np.log(odds_good)

s_va = to_score(p_va)
print(pd.Series(s_va).describe().round(1))


bands = pd.cut(s_va,bins=[-np.inf,500,520,540,560,580,np.inf])
band_table = (pd.DataFrame({"band":bands,"default":y_va.values})
              .groupby("band",observed =True)["default"]
              .agg(["mean","size"]))
print(band_table.round(3))

b0 = model.intercept_[0]
n = len(features)
bins = saved["bins"]

rows = []
for f in features:
    col = f.replace("woe_", "", 1)
    table = bins[col][0]
    b = coefs[f]
    for bin_label, w in table["woe"].items():
        pts = -factor * b * w + (offset - factor * b0) / n
        rows.append({"feature": col, "bin": bin_label,
                     "woe": round(w, 3), "points": round(pts)})

scorecard = pd.DataFrame(rows)
print(scorecard[scorecard["feature"] == "fico"])
scorecard.to_csv("data/processed/scorecard.csv", index=False)

first = valid.iloc[0] if "valid" in globals() else None
row = X_va.iloc[0]
manual = sum(-factor * coefs[f] * row[f] for f in features) + (offset - factor * b0)
print(f"Points total: {manual:.1f}   Model score: {s_va[0]:.1f}")

with open("data/processed/scorecard_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": features,
                 "factor": factor, "offset": offset}, f)
print("Saved scorecard model.")