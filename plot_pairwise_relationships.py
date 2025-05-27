import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

INPUT_PATH = "matched_data/steam_clean_preprocessed.json"
PLOT_DIR = "matched_data/"

VARS = [
    "user_reception_score",
    "violence_score",
    "no_sup_lang",
    "log_concurrent_users_yesterday",
    "log_price",
]


def load_data(input_path: str) -> pd.DataFrame:
    """Load preprocessed JSON data as DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def plot_pair(x: str, y: str, df: pd.DataFrame) -> None:
    """Plot scatterplot with regression line (LOWESS if continuous, else linear)."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y, data=df, alpha=0.3, label="Games")
    # Regression line
    if df[x].nunique() > 10:
        sns.regplot(x=x, y=y, data=df, scatter=False, lowess=True, line_kws={"color": "red", "lw": 2, "label": "LOWESS"})
    else:
        sns.regplot(x=x, y=y, data=df, scatter=False, lowess=False, line_kws={"color": "red", "lw": 2, "label": "Linear fit"})
    plt.title(f"{y} vs {x}")
    plt.legend()
    plt.tight_layout()
    out_path = f"{PLOT_DIR}scatter_{y}_vs_{x}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    """Plot pairwise relationships for all meaningful variable pairs."""
    df = load_data(INPUT_PATH)
    # Only use rows with nonzero values for both variables
    for i, x in enumerate(VARS):
        for y in VARS[i+1:]:
            sub = df[[x, y]].dropna()
            sub = sub[(sub[x] != 0) & (sub[y] != 0)]
            if len(sub) > 10:
                plot_pair(x, y, sub)
    # Optionally, pairplot for grid overview
    try:
        sub = df[VARS].replace(0, np.nan).dropna()
        if len(sub) > 10:
            sns.pairplot(sub, kind="scatter", plot_kws={"alpha": 0.3})
            plt.suptitle("Pairwise Relationships (nonzero only)", y=1.02)
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}pairplot_nonzero.png")
            plt.close()
            print(f"Saved {PLOT_DIR}pairplot_nonzero.png")
    except Exception as e:
        print(f"Could not create pairplot: {e}")

if __name__ == "__main__":
    main() 