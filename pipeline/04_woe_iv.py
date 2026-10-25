import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_parquet("data/processed/train.parquet")
valid = pd.read_parquet("data/processed/valid.parquet")
test = pd.read_parquet("data/processed/test.parquet")


def make_groups(x, edges):
    # Put each value into a bin (numbers) or keep its category (text); missing -> "Missing"
    g = pd.cut(x, edges) if edges is not None else x.astype("category")
    return g.cat.add_categories("Missing").fillna("Missing")


def woe_table(data, col, bins=10, eps=0.5):
    x = data[col]
    if pd.api.types.is_numeric_dtype(x):
        _, edges = pd.qcut(x, bins, retbins=True, duplicates="drop")
        if len(edges) < 4:
            vals = np.unique(x.dropna())[:5]
            edges = np.concatenate([[-np.inf], vals[:-1], [np.inf]])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = None

    t = data.groupby(make_groups(x, edges), observed=True)["default"].agg(bad="sum", total="size")
    # ... rest unchanged
    t.index = t.index.astype(str)
    t["good"] = t["total"] - t["bad"]
    t["bad_rate"] = t["bad"] / t["total"]

    dist_good = (t["good"] + eps) / t["good"].sum()
    dist_bad = (t["bad"] + eps) / t["bad"].sum()
    t["woe"] = np.log(dist_good / dist_bad)
    t["iv_part"] = (dist_good - dist_bad) * t["woe"]
    return t, edges, t["iv_part"].sum()


def apply_woe(data, col, table, edges):
    groups = make_groups(data[col], edges).astype(str)
    return groups.map(table["woe"]).fillna(0.0)


numeric = ["fico", "dti", "annual_inc", "loan_to_income", "credit_history_years",
           "emp_years", "revol_util", "revol_bal", "inq_last_6mths", "open_acc",
           "total_acc", "mort_acc", "delinq_2yrs", "pub_rec",
           "pub_rec_bankruptcies", "loan_amnt"]
categorical = ["home_ownership", "purpose", "verification_status",
               "application_type", "addr_state"]

bins_learned, ivs = {}, {}
for col in numeric + categorical:
    table, edges, iv = woe_table(train, col)
    bins_learned[col] = (table, edges)
    ivs[col] = iv

iv_series = pd.Series(ivs).sort_values(ascending=False)
print(iv_series.round(3))
print("pub_rec edges:", bins_learned["pub_rec"][1])
print(bins_learned["pub_rec"][0][["total", "bad_rate", "woe"]].round(3))
# Step 6: look inside the FICO table
print(bins_learned["fico"][0][["total", "bad_rate", "woe", "iv_part"]].round(3))

# Step 7: does each feature still work on 2014 data?
for col in iv_series.index[:8]:
    w = apply_woe(valid, col, *bins_learned[col])
    print(f"{col:22} IV={ivs[col]:.3f}  valid AUC={roc_auc_score(valid['default'], -w):.3f}")

# Step 8: build WoE datasets and save them
selected = list(iv_series[iv_series >= 0.02].index)
print("Selected:", selected)

def to_woe_frame(data):
    out = pd.DataFrame({f"woe_{c}": apply_woe(data, c, *bins_learned[c]) for c in selected})
    out["default"] = data["default"].values
    return out

to_woe_frame(train).to_parquet("data/processed/train_woe.parquet", index=False)
to_woe_frame(valid).to_parquet("data/processed/valid_woe.parquet", index=False)
to_woe_frame(test).to_parquet("data/processed/test_woe.parquet", index=False)

with open("data/processed/woe_bins.pkl", "wb") as f:
    pickle.dump({"bins": bins_learned, "selected": selected, "iv": ivs}, f)
print("Saved WoE datasets and bins.")