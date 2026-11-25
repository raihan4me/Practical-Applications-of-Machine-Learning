#!/usr/bin/env python3
"""Task 1: profile raw data, screen for leakage, write de-duplicated data."""
import pandas as pd
from sklearn.metrics import roc_auc_score

RAW = "data/raw/customers.csv"
CLEAN = "data/processed/customers_clean.csv"

# keep_default_na=False keeps the literal string "None" (a valid internet_service
# category) from being converted to NaN. Truly empty fields still become NaN
# via na_values=[""].
df = pd.read_csv(RAW, keep_default_na=False, na_values=[""])

print("== Shape ==")
print(df.shape)

print("\n== Dtypes ==")
print(df.dtypes.to_string())

print("\n== Missing values ==")
missing = df.isna().sum()
print(missing[missing > 0].to_string())

print("\n== Duplicates ==")
print(f"exact duplicate rows: {int(df.duplicated().sum())}")
print(f"duplicate customer_id: {int(df['customer_id'].duplicated().sum())}")

print("\n== Target balance ==")
print(df["churn"].value_counts(normalize=True).round(3).to_string())

print("\n== Churn rate by contract ==")
print(df.groupby("contract")["churn"].mean().round(3).sort_values(ascending=False).to_string())

print("\n== Single-feature AUC (leakage screen) ==")
numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "churn"]
suspects = []
for col in numeric_cols:
    x = df[col].fillna(df[col].median())
    auc = roc_auc_score(df["churn"], x)
    auc = max(auc, 1 - auc)
    flag = ""
    if auc > 0.90:
        flag = "  <-- SUSPICIOUS (possible leakage)"
        suspects.append(col)
    print(f"{col:22s} {auc:.3f}{flag}")

# Preserve the literal "None" string on write-back
clean = df.drop_duplicates().reset_index(drop=True)
clean.to_csv(CLEAN, index=False, na_rep="")
print(f"\nclean rows written: {len(clean)} -> {CLEAN}")
print(f"leakage suspects: {suspects}")
