import json
from typing import Dict, Any

FIELDS_TO_KEEP = [
    "name",
    "price",
    "supported_languages",
    "matched_reviews",
    "concurrent_users_yesterday",
    "violence_score",
    "positive_review_ratio",
    "user_reception_score",
    "no_sup_lang"
]

MATCHED_REVIEWS_FIELDS = [
    "review_score",
    "positive",
    "total"
]

def filter_game_fields(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter a game's dictionary to keep only the specified fields, and only selected fields in matched_reviews.

    Args:
        game: The original game dictionary.

    Returns:
        A filtered game dictionary.
    """
    filtered = {k: game[k] for k in FIELDS_TO_KEEP if k in game}
    # Filter matched_reviews if present
    if "matched_reviews" in filtered and isinstance(filtered["matched_reviews"], list):
        filtered_reviews = []
        for review in filtered["matched_reviews"]:
            filtered_review = {k: review[k] for k in MATCHED_REVIEWS_FIELDS if k in review}
            filtered_reviews.append(filtered_review)
        filtered["matched_reviews"] = filtered_reviews
    return filtered

def clean_steam_data(
    input_path: str,
    output_path: str
) -> None:
    """
    Cleans the dataset to keep only selected fields for each game.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    cleaned_data = {app_id: filter_game_fields(game) for app_id, game in data.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

def main() -> None:
    """
    Main function for CLI usage.
    """
    input_path = "matched_data/full_data.json"
    output_path = "matched_data/steam_clean.json"
    clean_steam_data(input_path, output_path)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main() 