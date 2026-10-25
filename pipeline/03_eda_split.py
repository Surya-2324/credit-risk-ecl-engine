import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_auc_score

df = pd.read_parquet("data/processed/loans_36m_2007_2015.parquet")
print(df.dtypes)

def to_number(s):
    if s.dtypes == "object":
        s = s.str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

df['int_rate'] = pd.to_numeric(df['int_rate'])
df["revol_util"] = to_number(df["revol_util"])
#  # Employment length: "10+ years" -> 10, "< 1 year" -> 0, missing stays missing

df["emp_years"] = (df["emp_length"]
                   .str.replace("< 1 year", "0", regex=False)
                   .str.extract(r"(\d+)")[0]
                   .astype(float))
# for how long the user has had the credit in years
df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"],
                                        format="%b-%Y", errors="coerce")
"""
In here we took the difference between the issue date and the earliest credit line to get the credit history in years. 
This is a useful feature for credit risk modeling, as it indicates how long the borrower has been using credit, 
which can be a predictor of their creditworthiness.
"""
df['credit_history_years'] = (df['issue_d'] - df["earliest_cr_line"]).dt.days/365.25
# One FICO number instead of a low/high range
df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

# Loan size relative to income
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)


train = df[df["year"] <= 2013].copy()
valid = df[df['year'] == 2014].copy()
test = df[df["year"] == 2015].copy()

print(train.shape[0])

for name, part in [("Train 2007-2013", train), ("Valid 2014", valid), ("Test 2015", test)]:
        print(f"{name:16} rows={len(part):>8,}  default rate={part['default'].mean():.2%}")

missing = train.isna().mean().sort_values(ascending=False)
print(missing[missing > 0].round(3))

cols = ["mort_acc", "emp_years", "revol_util", "pub_rec_bankruptcies"]
print(df[cols].isna().groupby(df['year']).mean().round(3))

def risk_by_bins(data, col, bins=10):
    groups = pd.qcut(data[col], bins, duplicates="drop")
    table = data.groupby(groups, observed=True)["default"].agg(["mean", "size"])
    miss = data[col].isna()
    if miss.any():
        table.loc["MISSING"] = [data.loc[miss, "default"].mean(), miss.sum()]
    return table.round(3)

for col in ["fico", "dti", "annual_inc", "loan_to_income", "credit_history_years", "emp_years"]:
    print(f"\n=== {col} ===")
    print(risk_by_bins(train, col))

for col in ["home_ownership", "purpose", "verification_status", "grade"]:
    print(f"\n=== {col} ===")
    print(train.groupby(col)["default"].agg(["mean", "size"])
               .sort_values("mean").round(3))

benchmark_auc = roc_auc_score(valid["default"], valid["int_rate"])
print(f"Lending Club interest rate as a PD score: AUC = {benchmark_auc:.3f}")

train.to_parquet("data/processed/train.parquet", index=False)
valid.to_parquet("data/processed/valid.parquet", index=False)
test.to_parquet("data/processed/test.parquet", index=False)
print("Saved train / valid / test.")