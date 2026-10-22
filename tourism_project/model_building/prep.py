"""Data preparation: load, clean, split and save train/test sets.

Outputs (repo root): Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv.
The workflow uploads these as the `data-splits` artifact for the training job.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")
TARGET = "ProdTaken"
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop columns that carry no predictive signal.
    #    "Unnamed: 0" is a leftover pandas index and CustomerID is a unique key.
    df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

    # 2. Fix a known typo in Gender ("Fe Male" -> "Female").
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

    # 3. Remove exact duplicate rows so the same record cannot sit in
    #    both train and test (that would inflate test scores).
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    # 4. Defensive imputation in case future data refreshes contain nulls.
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    df = clean(df)
    print(f"Cleaned shape: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train: {Xtrain.shape}, positive rate {ytrain.mean():.3f}")
    print(f"Test : {Xtest.shape}, positive rate {ytest.mean():.3f}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
