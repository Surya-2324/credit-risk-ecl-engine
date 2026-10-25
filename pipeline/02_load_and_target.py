import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = "data/raw/accepted_2007_to_2018Q4.csv.gz" 
peek = pd.read_csv(RAW, nrows=5)
print(peek.shape)
# print(list(peek.columns))
# print(peek.columns[:5])
print(peek["funded_amnt"])       


FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies", "application_type",
]
OUTCOMES = [
    "loan_status", "funded_amnt", "total_rec_prncp",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
]

# df = pd.read_csv(RAW, usecols = FEATURES + OUTCOMES, low_memory=False)
# print(df.shape)        

BAD = {"Charged Off", "Default",
       "Does not meet the credit policy. Status:Charged Off"}
GOOD = {"Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid"}

chunks =[]
status_counts = pd.Series(dtype="float64")

reader = pd.read_csv(RAW, usecols = FEATURES + OUTCOMES, low_memory=False, chunksize=100_000)

for i,chunk in enumerate(reader):
    # count every status filtering 
    status_counts = status_counts.add(chunk['loan_status'].value_counts(dropna =False), fill_value=0)

    # Keep only finished loans
    chunk = chunk[chunk['loan_status'].isin(BAD|GOOD)].copy()

    # converting floats to float32 to save memory
    floats = chunk.select_dtypes("float64").columns
    chunk[floats] = chunk[floats].astype("float32")

    chunks.append(chunk)
    print(f"Chunk {i}: kept {len(chunk):,}rows")
df = pd.concat(chunks, ignore_index=True)
del chunks
print(status_counts.sort_values(ascending=False).astype(int))

df['default'] = df['loan_status'].isin(BAD).astype(int)
print(f"Rows: {len(df):,}  Default rate: {df['default'].mean():.2%}")

#  Turning date and term into clean values
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
print(df['issue_d'].head())

df['term'] = df['term'].str.strip()

df['year'] = df['issue_d'].dt.year
print(df.groupby(["term", "year"])["default"]
        .agg(["mean", "size"]).round(3))


df = df[(df["term"] == "36 months") & (df["issue_d"] <= "2015-12-31")].copy()

print(f"Rows: {len(df):,}  Default rate: {df['default'].mean():.2%}")


for col in['recoveries','total_rec_prncp',"dti"]:
    score = df[col].fillna(0)
    auc = roc_auc_score(df['default'], score)
    print(f"{col:18} AUC = {max(auc, 1 -auc):.3f}")

import os

os.makedirs("data/processed", exist_ok=True)
df.to_parquet("data/processed/loans_36m_2007_2015.parquet", index=False)
print("Saved.")
