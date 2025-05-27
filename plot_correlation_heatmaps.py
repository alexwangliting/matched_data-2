import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

CSV_ALL = "matched_data/correlation_all_with_zeros.csv"
CSV_NONZERO = "matched_data/correlation_all_without_zeros.csv"


def build_matrix(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, list]:
    """Build a symmetric correlation matrix from long-form correlation DataFrame."""
    vars_ = sorted(set(df['var1']).union(df['var2']))
    mat = np.full((len(vars_), len(vars_)), np.nan)
    idx = {v: i for i, v in enumerate(vars_)}
    for _, row in df.iterrows():
        i, j = idx[row['var1']], idx[row['var2']]
        mat[i, j] = row[value_col]
        mat[j, i] = row[value_col]
        mat[i, i] = 1.0
        mat[j, j] = 1.0
    return mat, vars_


def plot_heatmap(mat: np.ndarray, vars_: list, title: str, out_path: str) -> None:
    """Plot and save a heatmap for the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=vars_, yticklabels=vars_, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    """Plot correlation heatmaps for all numeric variables, with and without zeros."""
    for csv, label in [(CSV_ALL, "with_zeros"), (CSV_NONZERO, "without_zeros")]:
        df = pd.read_csv(csv)
        for corr_type in ["pearson", "spearman"]:
            mat, vars_ = build_matrix(df, corr_type)
            title = f"{corr_type.capitalize()} Correlation ({label.replace('_', ' ')})"
            out_path = f"matched_data/heatmap_{corr_type}_{label}.png"
            plot_heatmap(mat, vars_, title, out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    main() 