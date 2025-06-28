import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from typing import List

INPUT_PATH = "04_FeatureEngineering_FeatureSelection/engineered_features.csv"
PLOT_DIR = "matched_data/"

MODELS = [
    {
        "outcome": "user_reception_score",
        "name": "User Reception",
        "features": ["metacritic_score", "n_tags", "price"],
    },
    {
        "outcome": "log_concurrent_users_yesterday",
        "name": "Player Engagement",
        "features": ["achievements", "n_tags", "metacritic_score", "no_sup_lang", "n_full_audio_lang", "price"],
    },
]

def load_data(input_path: str) -> pd.DataFrame:
    """Load engineered features as DataFrame."""
    return pd.read_csv(input_path)


def run_refined_regression(df: pd.DataFrame, y: str, X: List[str], model_name: str) -> None:
    """Run OLS regression with selected features, print summary, equation, and save residual plot."""
    sub = df[[y] + X].dropna()
    for col in [y] + X:
        sub = sub[sub[col] != 0]
    Y = sub[y]
    Xmat = sm.add_constant(sub[X])
    model = sm.OLS(Y, Xmat).fit()
    print(f"\n=== Refined Linear Regression Results: {model_name} ===")
    print(model.summary())
    # Save regression results to CSV
    results_df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "t-value": model.tvalues.values,
        "p-value": model.pvalues.values,
    })
    out_csv = f"05_multiple_linear_regression/Final Regression Results - {model_name}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved regression results to {out_csv}")
    # Print regression equation
    coefs = model.params
    eqn = f"{y} = {coefs[0]:.4f}"
    for i, feat in enumerate(X):
        eqn += f" + ({coefs[i+1]:.4f} * {feat})"
    print(f"\nRegression Equation for {model_name}:")
    print(eqn)
    # Residual plot
    plt.figure(figsize=(8, 6))
    plt.scatter(model.fittedvalues, model.resid, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residual Plot: {model_name} (Refined)")
    plt.tight_layout()
    out_path = f"{PLOT_DIR}residuals_{model_name.replace(' ', '_').lower()}_refined.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")
    # QQ plot
    plt.figure(figsize=(7, 7))
    sm.qqplot(model.resid, line='45', fit=True)
    plt.title(f"QQ Plot of Residuals: {model_name} (Refined)")
    plt.tight_layout()
    qq_out_path = f"{PLOT_DIR}qqplot_residuals_{model_name.replace(' ', '_').lower()}_refined.png"
    plt.savefig(qq_out_path)
    plt.close()
    print(f"Saved {qq_out_path}")


def main() -> None:
    """Run refined linear regression models for both outcomes using only significant features."""
    df = load_data(INPUT_PATH)
    for m in MODELS:
        run_refined_regression(df, m["outcome"], m["features"], m["name"])

if __name__ == "__main__":
    main() 