import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List
import os
from scipy.stats import skew, spearmanr, pearsonr

# Fields to analyze (from correlation_analysis.py)
CORRELATION_FIELDS = [
    "user_reception_score",
    "no_sup_lang",
    "violence_score",
    "price",
    "concurrent_users_yesterday",
]

INPUT_PATH = "matched_data/steam_clean.json"
REPORT_DIR = "matched_data/data_quality_report"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data(input_path: str) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    records = []
    for app_id, game in data.items():
        row = {field: game.get(field, None) for field in CORRELATION_FIELDS}
        records.append(row)
    return pd.DataFrame(records)


def check_missing_and_types(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Missing Values and Data Types ===")
    summary = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        dtype = df[col].dtype
        unique = df[col].nunique()
        print(f"{col}: {missing} missing, dtype={dtype}, unique={unique}")
        summary.append({"field": col, "missing": missing, "dtype": str(dtype), "unique": unique})
    return pd.DataFrame(summary)


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_distributions(df: pd.DataFrame):
    print("\n=== Plotting Distributions ===")
    for col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=50)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{REPORT_DIR}/{col}_hist.png")
        plt.close()
        # Boxplot for outliers
        plt.figure(figsize=(8, 2))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"{REPORT_DIR}/{col}_box.png")
        plt.close()


def check_skewness(df: pd.DataFrame):
    print("\n=== Skewness ===")
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 0:
            sk = skew(vals)
            print(f"{col}: skewness = {sk:.2f}")


def check_outliers(df: pd.DataFrame):
    print("\n=== Outlier Detection (IQR method) ===")
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 0:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((vals < lower) | (vals > upper)).sum()
            print(f"{col}: {outliers} outliers (IQR method)")


def check_discrete(df: pd.DataFrame):
    print("\n=== Discrete/Categorical Variable Check ===")
    for col in df.columns:
        vals = df[col].dropna()
        unique = np.unique(vals)
        if len(unique) < 10:
            print(f"{col}: likely discrete/categorical, unique values: {unique}")


def check_range_restriction(df: pd.DataFrame):
    print("\n=== Range Restriction ===")
    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"{col}: min={np.min(vals)}, max={np.max(vals)}, median={np.median(vals)}")


def check_sample_size(df: pd.DataFrame):
    print("\n=== Sample Size After Filtering ===")
    filtered = df.dropna()
    print(f"Rows with complete data: {len(filtered)} out of {len(df)}")
    return filtered


def check_multicollinearity(df: pd.DataFrame):
    print("\n=== Multicollinearity (Correlation Matrix) ===")
    corr = df.corr(method="pearson")
    print(corr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Pearson Correlation Matrix")
    plt.savefig(f"{REPORT_DIR}/correlation_matrix.png")
    plt.close()


def check_nonlinearity(df: pd.DataFrame):
    print("\n=== Nonlinearity (Spearman vs Pearson) ===")
    for i, col1 in enumerate(df.columns):
        for col2 in df.columns[i+1:]:
            vals = df[[col1, col2]].dropna()
            if len(vals) > 10:
                pearson = pearsonr(vals[col1], vals[col2])[0]
                spearman = spearmanr(vals[col1], vals[col2])[0]
                print(f"{col1} vs {col2}: Pearson={pearson:.3f}, Spearman={spearman:.3f}")


def main():
    df = load_data(INPUT_PATH)
    check_missing_and_types(df)
    df = convert_numeric(df)
    plot_distributions(df)
    check_skewness(df)
    check_outliers(df)
    check_discrete(df)
    check_range_restriction(df)
    filtered = check_sample_size(df)
    check_multicollinearity(filtered)
    check_nonlinearity(filtered)
    print(f"\nReport complete. Plots and outputs saved to {REPORT_DIR}/")

if __name__ == "__main__":
    main() 