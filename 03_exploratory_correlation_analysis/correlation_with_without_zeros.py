import json
import pandas as pd
import numpy as np
from typing import Any, Dict
from scipy.stats import pearsonr, spearmanr

INPUT_PATH = "matched_data/steam_clean_preprocessed.json"

FIELDS = [
    "user_reception_score",
    "no_sup_lang",
    "violence_score",
    "price",
    "concurrent_users_yesterday",
    "log_concurrent_users_yesterday",
]


def load_data(input_path: str) -> pd.DataFrame:
    """Load preprocessed JSON data as DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def report_zeros(df: pd.DataFrame, field: str) -> None:
    """Report the number and proportion of zeros in a field."""
    n_total = len(df)
    n_zeros = (df[field] == 0).sum()
    print(f"{field}: {n_zeros} zeros out of {n_total} rows ({n_zeros/n_total:.2%})")


def correlation_analysis(df: pd.DataFrame, x: str, y: str) -> None:
    """Compute and print Pearson and Spearman correlation between x and y."""
    vals = df[[x, y]].dropna()
    if len(vals) < 2:
        print(f"Not enough data for correlation between {x} and {y}.")
        return
    pearson = pearsonr(vals[x], vals[y])[0]
    spearman = spearmanr(vals[x], vals[y])[0]
    print(f"Correlation between {x} and {y} (n={len(vals)}): Pearson={pearson:.4f}, Spearman={spearman:.4f}")


def main() -> None:
    """Compare correlation results with and without zeros for concurrent users fields."""
    df = load_data(INPUT_PATH)
    for field in ["concurrent_users_yesterday", "log_concurrent_users_yesterday"]:
        print(f"\n=== Analysis for {field} ===")
        report_zeros(df, field)
        # With zeros
        print("With zeros:")
        correlation_analysis(df, "no_sup_lang", field)
        # Without zeros
        df_nonzero = df[df[field] != 0]
        print("Without zeros:")
        correlation_analysis(df_nonzero, "no_sup_lang", field)

if __name__ == "__main__":
    main() 