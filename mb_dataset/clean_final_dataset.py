import pandas as pd
import numpy as np

df = pd.read_csv("final_mobile_dataset.csv")

print("Before cleaning:", df.shape)

# -------------------------------
# REMOVE DUPLICATES (COMMENTED OUT)
# -------------------------------
# df = df.drop_duplicates()  # <--- Removes over half your dataset by mistake!

# -------------------------------
# FIX INF VALUES
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)

# -------------------------------
# 🔥 CLASS-WISE MEDIAN IMPUTATION
# -------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    df[col] = df.groupby("mapped_class")[col].transform(
        lambda x: x.fillna(x.median())
    )

# -------------------------------
# OPTIONAL: GLOBAL FILL (SAFETY)
# -------------------------------
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -------------------------------
# CLIP EXTREME VALUES
# -------------------------------
df[numeric_cols] = df[numeric_cols].clip(-1e5, 1e5)

# -------------------------------
# FINAL CHECK (NO DROPNA ❌)
# -------------------------------
print("\nRemaining NaNs:\n", df.isna().sum())

print("\nAfter cleaning:", df.shape)

# Save
df.to_csv("clean_mobile_dataset.csv", index=False)