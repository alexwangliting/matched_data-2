import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List

INPUT_PATH = "matched_data/steam_clean_preprocessed.json"
PLOT_DIR = "matched_data/"

VARS = [
    "user_reception_score",
    "log_concurrent_users_yesterday",
    "log_price",
    "violence_score",
    "no_sup_lang",
]


def load_data(input_path: str) -> pd.DataFrame:
    """Load preprocessed JSON data as DataFrame."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def run_regression(df: pd.DataFrame, y: str, X: List[str], model_name: str) -> None:
    """Run OLS regression, print summary, and save residual plot."""
    # Only use rows with nonzero values for all variables
    sub = df[[y] + X].dropna()
    for col in [y] + X:
        sub = sub[sub[col] != 0]
    Y = sub[y]
    Xmat = sm.add_constant(sub[X])
    model = sm.OLS(Y, Xmat).fit()
    print(f"\n=== Regression Results: {model_name} ===")
    print(model.summary())
    # Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot: {model_name}")
    plt.tight_layout()
    out_path = f"{PLOT_DIR}residuals_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    """Run multiple linear regression for user reception and player engagement."""
    df = load_data(INPUT_PATH)
    # Model 1: User Reception
    run_regression(
        df,
        y="user_reception_score",
        X=["log_price", "violence_score", "no_sup_lang"],
        model_name="User Reception"
    )
    # Model 2: Player Engagement (log concurrent users)
    run_regression(
        df,
        y="log_concurrent_users_yesterday",
        X=["log_price", "violence_score", "no_sup_lang"],
        model_name="Player Engagement"
    )

if __name__ == "__main__":
    main() 