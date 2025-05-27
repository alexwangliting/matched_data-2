# What Makes a Game Perform?  
**Analyzing the Impact of Price, Language Support, and Violence on Game Success**

## Overview

This project investigates how key features of PC games—**price, language availability, and violence level**—affect their performance on Steam, the world's largest digital distribution platform for PC games. We combine multiple public datasets, engineer relevant features, and apply robust statistical and machine learning methods to answer:  
**How do price, language support, and violence content influence user reception and player engagement?**

Our findings aim to inform game developers, publishers, and industry analysts about which factors most strongly drive a game's success.

---

## Table of Contents

- [Project Motivation & Research Questions](#project-motivation--research-questions)
- [Data Sources & Preparation](#data-sources--preparation)
- [Data Quality & Feature Engineering](#data-quality--feature-engineering)
- [Analytical Methods](#analytical-methods)
- [Key Results & Insights](#key-results--insights)
- [Visualizations](#visualizations)
- [How to Reproduce](#how-to-reproduce)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Motivation & Research Questions

### Objective

To quantify the impact of price, language support, and violence content on two key performance metrics for games:

- **User Reception:** Composite metric based on review scores and positive review ratios.
- **Player Engagement:** Measured by the number of concurrent users.

### Research Questions

1. **How does language support affect user reception and player engagement?**
2. **What is the effect of price and violence content on these outcomes?**
3. **Which features are the most robust predictors of game success?**

### Hypothesis

Games that are more affordable, available in multiple languages, and contain violent content are likely to perform better in terms of both user reception and engagement.

---

## Data Sources & Preparation

### Datasets Used

| Dataset                | Path                                      | Key Fields                                      | Size         |
|------------------------|-------------------------------------------|-------------------------------------------------|--------------|
| `games2.json`          | `01_original_datasets/games2.json`        | Game ID, price, supported languages, tags        | 97,410 games |
| `reviews.csv`          | `01_original_datasets/reviews.csv`        | Game ID, review score, positive/negative counts  | 140,154      |
| `steamspy_insights.csv`| `01_original_datasets/steamspy_insights.csv`| Game ID, concurrent users                        | 140,077      |

- **Merge Strategy:**  
  Datasets were merged on Game ID. Only games present in all three datasets were retained, ensuring complete data for all variables.

### Data Cleaning & Merging

- Scripts in `02_combining_preprocessing_datasets/` handle merging, cleaning, and feature creation:
  - `d1_match_reviews_with_games.py`
  - `d2_combine_concurrent_users.py`
  - `add_no_sup_lang.py`, `add_positive_review_ratio.py`, `add_user_reception_score.py`, `add_violence_score.py`
  - Output: `games_with_reviews_and_concurrent.json`, `steam_clean_preprocessed.json`

---

## Data Quality & Feature Engineering

### Coverage & Key Variables

- **Language Support:**  
  - 95.3% coverage; avg. 4.75 languages/game; English most common.
- **Violence Tags:**  
  - 95.2% coverage; 57.5% of games have violence-related tags.
  - Violence score computed from tags: Shooter, Horror, Combat, War, Violent, FPS, Psychological Horror, Gore.
- **Price:**  
  - 100% coverage; 20.25% of games are free-to-play.
- **Review Scores:**  
  - 94.5% coverage; avg. 4.22/9; 45.8% positive, 2.95% negative, 35% unrated.
- **Positive Review Ratio:**  
  - 87.5% coverage; strong correlation (r = 0.87) with review score.
- **Concurrent Users:**  
  - 94.5% coverage; 79.4% of games had 0 concurrent players.

### Feature Engineering

- **Composite Metrics:**  
  - *User Reception*: Combined review score and positive review ratio.
  - *Player Engagement*: Log-transformed concurrent users.
- **Additional Features:**  
  - Number of tags, genres, achievements, DLC count, game age (days), Metacritic score, number of supported languages (text/audio), violence score, price (log-transformed).

- **Zero Analysis:**  
  - Special handling for variables with high zero counts (e.g., concurrent users).
  - See: `count_zeros_per_variable.py`, `data_quality_report.py`, `preprocess_for_correlation.py`

---

## Analytical Methods

### Correlation Analysis

- **Pearson & Spearman correlations** to explore linear and monotonic relationships.
- Analysis performed both including and excluding games with zero concurrent users.
- Scripts:  
  - `03_exploratory_correlation_analysis/correlation_with_without_zeros.py`
  - `03_exploratory_correlation_analysis/correlation_with_without_zeros_allvars.py`
  - Results: `correlation_all_with_zeros.csv`, `correlation_all_without_zeros.csv`

### Regression Modeling

- **Multiple Linear Regression:**  
  - Baseline and refined models to estimate the effect of each feature on user reception and engagement.
  - Scripts:  
    - `05_multiple_linear_regression/multiple_linear_regression.py`
    - `05_multiple_linear_regression/multiple_linear_regression_rfe10.py`
    - `05_multiple_linear_regression/multiple_linear_regression_rfe10_refined.py`
- **Lasso Regression:**  
  - Regularized regression for feature selection and to prevent overfitting.
  - Cross-validation to select optimal regularization strength.
  - Script:  
    - `06_lasso_regression/lasso_regression_rfe10.py`

### Feature Selection

- **Variance Inflation Factor (VIF):**  
  - Removed features with high multicollinearity.
- **Recursive Feature Elimination (RFE):**  
  - Identified top predictors for each outcome.
  - Script:  
    - `04_FeatureEngineering_FeatureSelection/feature_selection_vif_rfe.py`
    - Results: `rfe_results.csv`, `vif_table.csv`

---

## Key Results & Insights

### Correlation Findings

- **Language Support & Player Engagement:**  
  - Weak correlation when including all games (many with zero players).
  - Moderate positive correlation (Pearson ≈ 0.25, Spearman ≈ 0.26) when focusing only on games with active players.
- **Language Support & User Reception:**  
  - Slight positive relationship, but weaker than for engagement.

### Regression Results

- **User Reception:**  
  - Best explained by Metacritic score, number of tags, and price (negative effect).
  - Language support is not a strong direct predictor.
  - Final model explains ~41% of variance (R² = 0.41).
- **Player Engagement:**  
  - Strongest predictors: achievements, Metacritic score, price, number of tags, and both text/audio language support.
  - Final model explains ~51% of variance (R² = 0.51).

### Lasso Regression

- **Feature Selection:**  
  - Lasso shrinks unimportant coefficients to zero, highlighting the most robust predictors.
  - For engagement: achievements, Metacritic score, price, tags, and language support are most important.
  - For user reception: Metacritic score and tags are most important; price has a small negative effect.

### Visual Diagnostics

- **Coefficient Path Plots:**  
  - Show which features remain important as regularization increases.
- **Predicted vs. Actual Plots:**  
  - Indicate reasonable model fit, with some scatter and outliers.
- **Residual Analysis:**  
  - Mild deviation from normality; outliers do not disproportionately affect regression.

---

## Visualizations

All visualizations are in `00_visualisation/`:

- **Correlation Heatmaps:**  
  - `00_visualisation/correlation_heatmaps/`
- **Tag/Violence Bar Charts:**  
  - `00_visualisation/tags_barchart/`
- **Skewness & Distribution Plots:**  
  - `00_visualisation/skewness_histograms/`
- **Lasso Coefficient Paths & Predicted vs. Actual:**  
  - `00_visualisation/lasso_plots/`
- **Residual Plots:**  
  - `00_visualisation/residual_plots/`

Scripts for generating plots are in the same directory, e.g., `plot_correlation_heatmaps.py`.

---

## How to Reproduce

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (see [Requirements](#requirements)).
3. **Run data preprocessing:**
   ```bash
   cd 02_combining_preprocessing_datasets
   python data_quality_report.py
   python preprocess_for_correlation.py
   # ...other scripts as needed
   ```
4. **Run feature engineering and selection:**
   ```bash
   cd ../04_FeatureEngineering_FeatureSelection
   python feature_engineering.py
   python feature_selection_vif_rfe.py
   ```
5. **Run analysis scripts:**
   ```bash
   cd ../03_exploratory_correlation_analysis
   python correlation_with_without_zeros.py
   python correlation_with_without_zeros_allvars.py

   cd ../05_multiple_linear_regression
   python multiple_linear_regression.py
   python multiple_linear_regression_rfe10.py
   python multiple_linear_regression_rfe10_refined.py

   cd ../06_lasso_regression
   python lasso_regression_rfe10.py
   ```
6. **View results** in the generated `.csv` and `.png` files in the respective folders.

---

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels
- (Optional) Jupyter Notebook for interactive exploration

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
00_visualisation/
    correlation_heatmaps/
    tags_barchart/
    skewness_histograms/
    lasso_plots/
    residual_plots/
    plot_correlation_heatmaps.py
01_original_datasets/
    games2.json
    reviews.csv
    steamspy_insights.csv
02_combining_preprocessing_datasets/
    *.py
    *.json
03_exploratory_correlation_analysis/
    *.py
    *.csv
04_FeatureEngineering_FeatureSelection/
    *.py
    *.csv
    *.json
05_multiple_linear_regression/
    *.py
06_lasso_regression/
    *.py
```

---

## Acknowledgements

- Steam for providing the data via their API and SteamSpy.
- Kaggle and Steam Insights for public datasets.
- scikit-learn and statsmodels for modeling tools.
- matplotlib and seaborn for visualization.

---

## License

MIT License

---

## Contact

For questions or collaboration, please open an issue or contact [Your Name](mailto:your.email@example.com).

---

**If you use this project or its findings, please cite this repository!**

---

Let me know if you want a shorter version, want to add badges, or need a requirements.txt template! 