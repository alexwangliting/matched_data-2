import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List

INPUT_PATH = "matched_data/engineered_features.csv"
PLOT_DIR = "matched_data/"

OUTCOMES = [
    ("user_reception_score", "User Reception"),
    ("log_concurrent_users_yesterday", "Player Engagement"),
]
PREDICTORS = [
    "game_age_days", "dlc_count", "achievements", "is_multiplayer", "n_categories", "n_genres", "n_tags",
    "review_count", "metacritic_score", "estimated_owners", "windows", "mac", "linux", "no_sup_lang",
    "n_full_audio_lang", "price", "violence_score"
]

def load_data(input_path: str) -> pd.DataFrame:
    """Load engineered features as DataFrame."""
    return pd.read_csv(input_path)


def run_regression(df: pd.DataFrame, y: str, X: List[str], model_name: str) -> None:
    """Run OLS regression, print summary, save residual plot."""
    sub = df[[y] + X].dropna()
    for col in [y] + X:
        sub = sub[sub[col] != 0]
    Y = sub[y]
    Xmat = sm.add_constant(sub[X])
    model = sm.OLS(Y, Xmat).fit()
    print(f"\n=== Linear Regression Results: {model_name} ===")
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
    """Run and compare linear regression models for both outcomes."""
    df = load_data(INPUT_PATH)
    for y, model_name in OUTCOMES:
        run_regression(df, y, PREDICTORS, model_name)

if __name__ == "__main__":
    main() 