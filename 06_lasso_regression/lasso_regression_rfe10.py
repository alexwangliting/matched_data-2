import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List
import os

INPUT_PATH = "04_FeatureEngineering_FeatureSelection/engineered_features.csv"
PLOT_DIR = "matched_data/"

OUTCOMES = [
    ("user_reception_score", "User Reception"),
    ("log_concurrent_users_yesterday", "Player Engagement"),
]
RFE10_FEATURES = [
    "game_age_days", "dlc_count", "achievements", "n_genres", "n_tags", "metacritic_score",
    "no_sup_lang", "n_full_audio_lang", "price", "violence_score"
]


def load_data(input_path: str) -> pd.DataFrame:
    """Load engineered features as DataFrame."""
    return pd.read_csv(input_path)


def run_lasso(df: pd.DataFrame, y: str, X: List[str], model_name: str) -> None:
    """Run LassoCV, print results, plot coefficient path and pred vs actual."""
    sub = df[[y] + X].dropna()
    for col in [y] + X:
        sub = sub[sub[col] != 0]
    Y = sub[y].values
    Xmat = sub[X].values
    scaler = StandardScaler()
    Xmat_scaled = scaler.fit_transform(Xmat)
    X_train, X_test, y_train, y_test = train_test_split(Xmat_scaled, Y, test_size=0.2, random_state=42)
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, y_train)
    print(f"\n=== Lasso Regression Results: {model_name} ===")
    print(f"Best alpha: {lasso.alpha_:.4f}")
    print("Coefficients:")
    for feat, coef in zip(X, lasso.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"Train R^2: {lasso.score(X_train, y_train):.3f} | Test R^2: {lasso.score(X_test, y_test):.3f}")

    # Save results as CSV
    os.makedirs(PLOT_DIR, exist_ok=True)
    results = pd.DataFrame({
        'feature': X,
        'coefficient': lasso.coef_
    })
    results['best_alpha'] = lasso.alpha_
    results['train_r2'] = lasso.score(X_train, y_train)
    results['test_r2'] = lasso.score(X_test, y_test)
    csv_out_path = f"{PLOT_DIR}lasso_results_{model_name.replace(' ', '_').lower()}.csv"
    results.to_csv(csv_out_path, index=False)
    print(f"Saved {csv_out_path}")

    # Coefficient path plot
    plt.figure(figsize=(8, 6))
    m_log_alphas = -np.log10(lasso.alphas_)
    plt.title(f"Lasso Coefficient Paths: {model_name}")
    # Remove previous labeling attempts
    # plt.tight_layout()
    # plt.legend(X, loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    # coef_paths = lasso.path(X_train, y_train, alphas=lasso.alphas_, max_iter=10000)[1].T
    # for i, feature in enumerate(X):
    #     plt.text(m_log_alphas[-1], coef_paths[-1, i], feature, fontsize='small', ha='left', va='center')

    # Plot paths and add labels to the right
    ax = plt.gca()
    lines = ax.plot(m_log_alphas, lasso.path(X_train, y_train, alphas=lasso.alphas_, max_iter=10000)[1].T)
    ax.set_xlabel("-log10(alpha)")
    ax.set_ylabel("Coefficients")
    ax.set_title(f"Lasso Coefficient Paths: {model_name}")

    # Add text labels to the right of the plot
    # Sort features based on their final coefficient values to order labels vertically
    final_coefs = [line.get_ydata()[-1] for line in lines]
    sorted_features = [feature for _, feature in sorted(zip(final_coefs, X))]

    for i, feature in enumerate(sorted_features):
        # Find the corresponding line to get its last position and color
        line_index = X.index(feature)
        line = lines[line_index]
        y_pos = line.get_ydata()[-1] # Get the last y-coordinate of the line
        x_pos = m_log_alphas[-1] + 0.05 # Add a small offset
        color = line.get_color()

        # Add a small vertical offset based on the sorted index to prevent overlap
        vertical_offset = (i - len(sorted_features) / 2) * 0.003 # Adjust the multiplier (0.003) as needed

        ax.text(x_pos, y_pos + vertical_offset, feature, fontsize='small', ha='left', va='center', color=color)

    # Adjust the right margin to make space for labels and increase figure size slightly
    plt.subplots_adjust(right=0.7)
    plt.gcf().set_size_inches(10, 6)

    out_path = f"{PLOT_DIR}lasso_coef_path_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")
    # Predicted vs actual plot
    y_pred = lasso.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Lasso: {model_name}\nPredicted vs Actual (Test Set)")
    plt.tight_layout()
    out_path = f"{PLOT_DIR}lasso_pred_vs_actual_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    """Run Lasso regression for both outcomes using RFE top 10 features."""
    df = load_data(INPUT_PATH)
    for y, model_name in OUTCOMES:
        run_lasso(df, y, RFE10_FEATURES, model_name)

if __name__ == "__main__":
    main() 