import pandas as pd
import glob
import os

# Path to datasets folder
files = glob.glob("datasets/*.csv")

print("Cleaning files:", files)

for file in files:
    df = pd.read_csv(file)

    print(f"\nProcessing: {file}")
    print("Before:", df.shape)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove obvious invalid values (example)
    if "typing_speed" in df.columns:
        df = df[(df["typing_speed"] >= 0) | (df["typing_speed"].isna())]

    # Save cleaned version (overwrite or new file)
    filename = os.path.basename(file)
    df.to_csv(f"datasets/cleaned_{filename}", index=False)

    print("After:", df.shape)

print("\nAll files cleaned.")