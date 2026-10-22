"""Register the tourism dataset that lives inside the GitHub repository.

The CSV is version-controlled in tourism_project/data/, so "registration" means:
check the file is present, validate the schema, and print a short summary
that shows up in the GitHub Actions log for this job.
"""
import hashlib
import os
import sys

import pandas as pd

DATA_PATH = os.path.join("tourism_project", "data", "tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing expected columns: {missing}")

    if not set(df["ProdTaken"].unique()) <= {0, 1}:
        sys.exit("Target column ProdTaken must be binary (0/1)")

    with open(DATA_PATH, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    print("Dataset registered from repository")
    print(f"  path     : {DATA_PATH}")
    print(f"  sha256   : {sha256}")
    print(f"  rows     : {df.shape[0]}")
    print(f"  columns  : {df.shape[1]}")
    print(f"  missing  : {int(df.isna().sum().sum())} null values")
    print("  target   : ProdTaken")
    print(df["ProdTaken"].value_counts().rename("count").to_string())


if __name__ == "__main__":
    main()
