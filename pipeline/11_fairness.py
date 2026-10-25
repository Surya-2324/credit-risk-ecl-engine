import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import brentq
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
with open("data/processed/lgbm_model.pkl", "rb") as f:
    saved = pickle.load(f)
model, FEATS, CATEGORICAL = saved["model"], saved["features"], saved["categorical"]

for c in CATEGORICAL:
    cats = train[c].astype("category").cat.categories
    for d in (train, valid):
        d[c] = pd.Categorical(d[c].where(d[c].isin(cats)), categories=cats)

REGION = {}
for st in "CT ME MA NH RI VT NJ NY PA".split():
    REGION[st] = "Northeast"
for st in "IL IN MI OH WI IA KS MN MO NE ND SD".split():
    REGION[st] = "Midwest"
for st in "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX".split():
    REGION[st] = "South"
for st in "AZ CO ID MT NV NM UT WY AK CA HI OR WA".split():
    REGION[st] = "West"

valid["region"] = valid["addr_state"].astype(str).map(REGION).fillna("Other")
print(valid["region"].value_counts())

feats_ns = [f for f in FEATS if f != "addr_state"]
cons_all = model.get_params()["monotone_constraints"]
cons_ns = [c for f, c in zip(FEATS, cons_all) if f != "addr_state"]

params = model.get_params()
params.update(monotone_constraints=cons_ns, n_estimators=model.best_iteration_)
m_ns = lgb.LGBMClassifier(**params)
m_ns.fit(train[feats_ns], train["default"])

y = valid["default"].values
raw_with = model.predict_proba(valid[FEATS])[:, 1]
raw_without = m_ns.predict_proba(valid[feats_ns])[:, 1]

print(f"AUC with state:    {roc_auc_score(y, raw_with):.3f}")
print(f"AUC without state: {roc_auc_score(y, raw_without):.3f}")

def recalibrate(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    d = brentq(lambda s: expit(logit(p) + s).mean() - y.mean(), -3, 3)
    return expit(logit(p) + d)

pd_with = recalibrate(raw_with, y)
pd_without = recalibrate(raw_without, y)

cutoff_with = np.quantile(pd_with, 0.80)
cutoff_without = np.quantile(pd_without, 0.80)
approve_with = pd_with <= cutoff_with
approve_without = pd_without <= cutoff_without

def fairness_table(pd_vals, approve, name):
    df = pd.DataFrame({"region": valid["region"].values, "y": y,
                       "pd": pd_vals, "approve": approve})
    t = df.groupby("region").agg(n=("y", "size"),
                                 actual=("y", "mean"),
                                 mean_pd=("pd", "mean"),
                                 approval=("approve", "mean"))
    t["AIR"] = t["approval"] / t["approval"].max()
    t["false_decline"] = df[df["y"] == 0].groupby("region")["approve"].apply(lambda a: 1 - a.mean())
    t["missed_default"] = df[df["y"] == 1].groupby("region")["approve"].mean()
    print(f"\n=== {name} ===")
    print(t.round(3))
    return t

t_with = fairness_table(pd_with, approve_with, "With addr_state")
t_without = fairness_table(pd_without, approve_without, "Without addr_state")

valid["state_shift"] = logit(np.clip(raw_with, 1e-6, 1 - 1e-6)) - logit(np.clip(raw_without, 1e-6, 1 - 1e-6))

print("\nAverage shift in log-odds caused by state, by region:")
print(valid.groupby("region")["state_shift"].agg(["mean", "std"]).round(3))

by_state = (valid.groupby("addr_state", observed=True)["state_shift"]
                 .agg(["mean", "size"]).query("size >= 500").sort_values("mean"))
print("\nStates where location LOWERS predicted risk most:")
print(by_state.head(5).round(3))
print("\nStates where location RAISES predicted risk most:")
print(by_state.tail(5).round(3)) 

auc_with = roc_auc_score(y, raw_with)
auc_without = roc_auc_score(y, raw_without)
min_air = min(t_with["AIR"].min(), t_without["AIR"].min())

card = f"""# Model Card: Lending Club PD Model

## Model details
- **Type:** LightGBM gradient boosting with monotonic constraints, plus Platt/intercept calibration.
- **Champion/challenger:** WoE logistic-regression scorecard (validation AUC 0.645) as the explainable challenger.
- **Target:** default within the loan term (Charged Off / Default = 1, Fully Paid = 0).

## Intended use
- Estimating probability of default for 36-month unsecured personal loans at application.
- Inputs to approval decisions, expected-loss estimation, and pricing review.
- **Not intended for:** other loan products, terms, or markets without revalidation.

## Data
- Lending Club accepted loans, 36-month term, issued 2007–2015.
- Out-of-time split: train 2007–2013 (175,426 loans, 12.63% default), validation 2014 (162,570, 13.73%), test 2015 (283,026, 14.89%, held out).
- Unfinished loans excluded; vintages after 2015 excluded to avoid maturity bias.

## Features
- Application-time features only (whitelist): credit score, DTI, income, utilisation, inquiries, credit history, and others.
- **Excluded:** post-origination columns (leakage), and Lending Club grade / interest rate (another lender's model; used as a benchmark).
- Monotonic constraints on features with clear direction (e.g. higher FICO never increases PD).

## Performance (validation 2014)
- LightGBM AUC: {auc_with:.3f} (Gini {2*auc_with - 1:.3f}); Lending Club interest-rate benchmark: 0.658.
- Monotonic constraints cost about 0.001 AUC versus an unconstrained model.
- Without `addr_state`: AUC {auc_without:.3f}.

## Calibration
- Uncalibrated PDs under-predicted 2014 defaults by about 4% because of population drift.
- Class rebalancing (SMOTE-style) left AUC unchanged but inflated mean PD from 14% to 47%; not used.
- Calibration refreshed on the most recent year; monitored over time.

## Fairness
- No protected attributes available; geography tested as a proxy across US Census regions.
- Lowest Adverse Impact Ratio across regions: {min_air:.2f} (threshold for review: 0.80).
- Decision on `addr_state`: [WRITE YOUR DECISION AND REASON HERE].
- Reason codes exclude state of residence.
- A production deployment would add proxy-based testing (e.g. BISG).

## Limitations
- Only approved loans are observed (no reject inference), so risk among rejected applicants is unknown.
- Population drift: default rates rose from 12.6% to 14.9% across the split; calibration requires regular refresh.
- Profit analysis assumes full-term interest and ignores funding and operating costs.

## Monitoring plan
- Population Stability Index (PSI) on scores and key features; calibration traffic lights by decile; quarterly review.
"""

with open("reports/model_card.md", "w", encoding="utf-8") as f:
    f.write(card)
print("\nSaved reports/model_card.md")