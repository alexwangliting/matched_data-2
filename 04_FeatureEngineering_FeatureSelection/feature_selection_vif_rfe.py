import pandas as pd
import numpy as np
from typing import List
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

INPUT_PATH = "matched_data/engineered_features.csv"
VIF_OUT = "matched_data/vif_table.csv"
RFE_OUT = "matched_data/rfe_results.csv"

PREDICTORS = [
    "game_age_days", "dlc_count", "achievements", "is_multiplayer", "n_categories", "n_genres", "n_tags",
    "review_count", "metacritic_score", "estimated_owners", "windows", "mac", "linux", "no_sup_lang",
    "n_full_audio_lang", "price", "violence_score"
]
OUTCOMES = [
    ("user_reception_score", "User Reception"),
    ("log_concurrent_users_yesterday", "Player Engagement"),
]

def calculate_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate VIF for each feature."""
    X = df[features].dropna()
    vif_data = []
    for i, col in enumerate(features):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": col, "vif": vif})
    vif_df = pd.DataFrame(vif_data)
    return vif_df

def run_rfe(df: pd.DataFrame, y: str, X: List[str], n_features: int) -> List[str]:
    """Run RFE to select top n_features."""
    sub = df[[y] + X].dropna()
    for col in [y] + X:
        sub = sub[sub[col] != 0]
    Xmat = sub[X].values
    Y = sub[y].values
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(Xmat, Y)
    selected = [f for f, s in zip(X, rfe.support_) if s]
    return selected

def main() -> None:
    """VIF diagnostics and RFE feature selection for engineered features."""
    df = pd.read_csv(INPUT_PATH)
    # 1. VIF
    vif_df = calculate_vif(df, PREDICTORS)
    print("\n=== VIF Table ===")
    print(vif_df)
    vif_df.to_csv(VIF_OUT, index=False)
    # 2. Remove features with VIF > 5
    reduced = vif_df[vif_df["vif"] <= 5]["feature"].tolist()
    print(f"\nFeatures with VIF <= 5: {reduced}")
    # 3. RFE for each outcome
    rfe_results = []
    for y, outcome in OUTCOMES:
        for n in [5, 10]:
            selected = run_rfe(df, y, reduced, n)
            print(f"\nRFE Top {n} Features for {outcome}: {selected}")
            rfe_results.append({"outcome": outcome, "n_features": n, "selected_features": selected})
    pd.DataFrame(rfe_results).to_csv(RFE_OUT, index=False)
    print(f"\nSaved {VIF_OUT} and {RFE_OUT}")

if __name__ == "__main__":
    main() 