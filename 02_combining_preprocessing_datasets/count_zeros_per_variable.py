import json
import pandas as pd
from typing import Any, Dict

INPUT_PATH = "matched_data/steam_clean_preprocessed.json"


def load_data(input_path: str) -> pd.DataFrame:
    """Load preprocessed JSON data as DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def count_zeros(df: pd.DataFrame) -> None:
    """Print the number and proportion of zeros for each column in the DataFrame."""
    n_total = len(df)
    print(f"{'Variable':<35} {'# Zeros':>10} {'% Zeros':>10}")
    print("-" * 60)
    for col in df.columns:
        n_zeros = (df[col] == 0).sum()
        pct_zeros = n_zeros / n_total * 100
        print(f"{col:<35} {n_zeros:>10} {pct_zeros:>9.2f}%")


def main() -> None:
    """Count and print zeros for each variable in the preprocessed data."""
    df = load_data(INPUT_PATH)
    count_zeros(df)

if __name__ == "__main__":
    main() 