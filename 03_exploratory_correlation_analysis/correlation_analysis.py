import json
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

CORRELATION_PAIRS: List[Tuple[str, str]] = [
    ("user_reception_score", "no_sup_lang"),
    ("user_reception_score", "violence_score"),
    ("user_reception_score", "price"),
    ("user_reception_score", "concurrent_users_yesterday"),
    ("concurrent_users_yesterday", "no_sup_lang"),
    ("concurrent_users_yesterday", "violence_score"),
    ("concurrent_users_yesterday", "price"),
]

FIELDS = set([f for pair in CORRELATION_PAIRS for f in pair])

INPUT_PATH = "matched_data/steam_clean.json"
OUTPUT_PATH = "matched_data/correlation_results.txt"

def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the JSON data and return a DataFrame with relevant fields.

    Args:
        input_path: Path to the input JSON file.
    Returns:
        DataFrame with relevant fields.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)
    records = []
    for app_id, game in data.items():
        row = {field: game.get(field, None) for field in FIELDS}
        records.append(row)
    return pd.DataFrame(records)

def check_nulls(df: pd.DataFrame) -> None:
    """
    Print a summary of null values in the DataFrame.

    Args:
        df: DataFrame to check for nulls.
    """
    print("Null Value Summary (before filtering):")
    print("==============================")
    null_counts = df.isnull().sum()
    total_nulls = df.isnull().sum().sum()
    for col, count in null_counts.items():
        print(f"{col}: {count} null values")
    print(f"Total null values in dataset: {total_nulls}\n")

def filter_numeric_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows to keep only those where all fields are present and numeric.

    Args:
        df: DataFrame with possible nulls or non-numeric values.
    Returns:
        Filtered DataFrame with only numeric values.
    """
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

def compute_correlations(df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    """
    Compute Pearson correlation for each specified pair.

    Args:
        df: DataFrame with relevant fields.
    Returns:
        Dictionary mapping field pairs to correlation coefficients.
    """
    results = {}
    for f1, f2 in CORRELATION_PAIRS:
        if f1 in df.columns and f2 in df.columns:
            corr = df[f1].corr(df[f2])
            results[(f1, f2)] = corr
        else:
            results[(f1, f2)] = np.nan
    return results

def save_results(results: Dict[Tuple[str, str], float], output_path: str) -> None:
    """
    Save correlation results to a text file.

    Args:
        results: Dictionary of correlation results.
        output_path: Path to save the results.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Correlation Analysis Results\n")
        f.write("==========================\n")
        for (f1, f2), corr in results.items():
            line = f"Correlation between '{f1}' and '{f2}': {corr:.4f}\n"
            f.write(line)

def main() -> None:
    """
    Main function to run correlation analysis.
    """
    df = load_data(INPUT_PATH)
    check_nulls(df)
    df_filtered = filter_numeric_rows(df)
    results = compute_correlations(df_filtered)
    print("Correlation Analysis Results")
    print("==========================")
    for (f1, f2), corr in results.items():
        print(f"Correlation between '{f1}' and '{f2}': {corr:.4f}")
    save_results(results, OUTPUT_PATH)
    print(f"\nResults saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 