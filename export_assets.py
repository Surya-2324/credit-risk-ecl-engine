"""Bundle the trained models and report files into dashboard/assets for the Streamlit app.

Run from the project folder (D:\\Credit_Risk):
    python export_assets.py
"""
import os
import pickle
import shutil

import pandas as pd


def load(name):
    with open(f"data/processed/{name}.pkl", "rb") as f:
        return pickle.load(f)


lg_pd = load("lgbm_model")
train = pd.read_parquet("data/processed/train.parquet")

FEATS, CATEGORICAL = lg_pd["features"], lg_pd["categorical"]
NUMERIC = [f for f in FEATS if f not in CATEGORICAL]

# Category lists exactly as the model saw them in training
categories = {c: [str(v) for v in train[c].astype("category").cat.categories] for c in CATEGORICAL}
cat_defaults = {c: str(train[c].mode()[0]) for c in CATEGORICAL}

# Typical values to pre-fill the input form
defaults = {f: float(train[f].median()) for f in NUMERIC}

bundle = {
    "pd_model": lg_pd["model"],
    "features": FEATS,
    "categorical": CATEGORICAL,
    "categories": categories,
    "cat_defaults": cat_defaults,
    "defaults": defaults,
    "calibrator": load("calibrator"),
    "lgd": load("lgd_model"),
    "ead": load("ead_model"),
    "reasons": load("reason_templates"),
}

os.makedirs("dashboard/assets/figures", exist_ok=True)
with open("dashboard/assets/bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Aggregated results and charts only; no loan-level data is copied
for src in ["reports/ecl_summary.csv", "reports/model_card.md", "reports/monitoring_report.csv"]:
    if os.path.exists(src):
        shutil.copy(src, "dashboard/assets/")
if os.path.isdir("reports/figures"):
    for name in os.listdir("reports/figures"):
        if name.endswith(".png"):
            shutil.copy(os.path.join("reports/figures", name), "dashboard/assets/figures/")

size_mb = os.path.getsize("dashboard/assets/bundle.pkl") / 1e6
print(f"Saved dashboard/assets/bundle.pkl ({size_mb:.1f} MB) and copied reports.")
