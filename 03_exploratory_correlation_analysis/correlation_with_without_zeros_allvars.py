import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
import itertools

INPUT_PATH = "matched_data/steam_clean_preprocessed.json"
CSV_ALL = "matched_data/correlation_all_with_zeros.csv"
CSV_NONZERO = "matched_data/correlation_all_without_zeros.csv"

# Define which variables to use
BASE_VARS = ["user_reception_score", "violence_score", "no_sup_lang"]
LOG_VARS = ["log_concurrent_users_yesterday", "log_price"]
ALL_VARS = BASE_VARS + LOG_VARS


def is_log_pair(x: str, y: str) -> bool:
    """Return True if x and y are a variable and its log-transformed version."""
    if x.startswith("log_") and x[4:] == y:
        return True
    if y.startswith("log_") and y[4:] == x:
        return True
    return False


def load_data(input_path: str) -> pd.DataFrame:
    """Load preprocessed JSON data as DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_correlations(df: pd.DataFrame, pairs: List[Tuple[str, str]], exclude_zeros: bool = False) -> List[Dict[str, Any]]:
    """Compute Pearson and Spearman correlations for all pairs, optionally excluding zeros."""
    results = []
    for x, y in pairs:
        if is_log_pair(x, y):
            continue
        sub = df[[x, y]].dropna()
        if exclude_zeros:
            sub = sub[(sub[x] != 0) & (sub[y] != 0)]
        n = len(sub)
        if n < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = pearsonr(sub[x], sub[y])[0]
            spearman = spearmanr(sub[x], sub[y])[0]
        results.append({
            "var1": x,
            "var2": y,
            "n": n,
            "pearson": pearson,
            "spearman": spearman,
            "exclude_zeros": exclude_zeros
        })
    return results


def main() -> None:
    """Compute pairwise correlations for meaningful variable pairs only, with and without zeros."""
    df = load_data(INPUT_PATH)
    # Only use the selected variables
    available_vars = [v for v in ALL_VARS if v in df.columns]
    pairs = [(x, y) for x, y in itertools.combinations(available_vars, 2)]

    print(f"\n=== Correlations INCLUDING zeros ===")
    results_with_zeros = compute_correlations(df, pairs, exclude_zeros=False)
    df_with_zeros = pd.DataFrame(results_with_zeros)
    print(df_with_zeros[['var1', 'var2', 'n', 'pearson', 'spearman']])
    df_with_zeros.to_csv(CSV_ALL, index=False)

    print(f"\n=== Correlations EXCLUDING zeros ===")
    results_without_zeros = compute_correlations(df, pairs, exclude_zeros=True)
    df_without_zeros = pd.DataFrame(results_without_zeros)
    print(df_without_zeros[['var1', 'var2', 'n', 'pearson', 'spearman']])
    df_without_zeros.to_csv(CSV_NONZERO, index=False)

    print(f"\nResults saved to {CSV_ALL} and {CSV_NONZERO}")

if __name__ == "__main__":
    main() 