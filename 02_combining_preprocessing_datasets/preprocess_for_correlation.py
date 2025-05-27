import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List
import os

CORRELATION_FIELDS = [
    "user_reception_score",
    "no_sup_lang",
    "violence_score",
    "price",
    "concurrent_users_yesterday",
]

INPUT_PATH = "matched_data/steam_clean.json"
CSV_OUTPUT = "matched_data/steam_clean_preprocessed.csv"
JSON_OUTPUT = "matched_data/steam_clean_preprocessed.json"

SKEWED_FIELDS = ["concurrent_users_yesterday", "price"]
VARIANCE_THRESHOLD = 1e-6  # Remove fields with variance below this


def load_data(input_path: str) -> pd.DataFrame:
    """Load JSON data and return DataFrame with relevant fields."""
    with open(input_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    records = []
    for app_id, game in data.items():
        row = {field: game.get(field, None) for field in CORRELATION_FIELDS}
        records.append(row)
    return pd.DataFrame(records)


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns to numeric, coercing errors to NaN."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def log_transform(df: pd.DataFrame, fields: List[str]) -> pd.DataFrame:
    """Apply log1p transformation to specified fields."""
    for field in fields:
        if field in df.columns:
            df[f"log_{field}"] = np.log1p(df[field])
    return df


def remove_low_variance(df: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    """Remove columns with variance below threshold."""
    low_var_cols = [col for col in df.columns if df[col].var(skipna=True) < threshold]
    if low_var_cols:
        print(f"Removing low-variance columns: {low_var_cols}")
        df = df.drop(columns=low_var_cols)
    return df


def encode_discrete(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure discrete/categorical fields are correctly encoded."""
    # violence_score is likely discrete
    if "violence_score" in df.columns:
        unique = df["violence_score"].dropna().unique()
        if len(unique) < 10:
            df["violence_score"] = df["violence_score"].astype("Int64")
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in any correlation field."""
    before = len(df)
    df = df.dropna(subset=CORRELATION_FIELDS)
    after = len(df)
    print(f"Dropped {before - after} rows due to missing values. Remaining: {after}")
    return df


def report_stats(df: pd.DataFrame) -> None:
    """Print sample size and basic statistics."""
    print(f"\nSample size after preprocessing: {len(df)}")
    print("Basic statistics:")
    print(df.describe(include="all").T)


def save_outputs(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    """Save DataFrame to CSV and JSON."""
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"Saved preprocessed data to {csv_path} and {json_path}")


def main() -> None:
    """Preprocess steam_clean.json for correlation analysis."""
    df = load_data(INPUT_PATH)
    df = convert_numeric(df)
    df = log_transform(df, SKEWED_FIELDS)
    df = encode_discrete(df)
    df = remove_low_variance(df, VARIANCE_THRESHOLD)
    df = drop_missing(df)
    report_stats(df)
    save_outputs(df, CSV_OUTPUT, JSON_OUTPUT)

if __name__ == "__main__":
    main() 