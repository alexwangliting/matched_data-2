import json
from typing import Dict, Any, Optional

def calculate_user_reception_score(
    positive_review_ratio: Optional[float],
    matched_reviews: list[dict[str, Any]]
) -> Optional[float]:
    """
    Calculate the user reception score as the average of positive_review_ratio and review_score.

    Args:
        positive_review_ratio: The positive review ratio as a float or None.
        matched_reviews: List of review dicts, each with 'review_score' as a string or number.

    Returns:
        The user reception score as a float, or None if not computable.
    """
    if not matched_reviews or positive_review_ratio is None:
        return None
    review = matched_reviews[0]
    try:
        review_score = float(review.get('review_score'))
        normalized_review_score = review_score / 10  # Normalize to 0-1 scale
        return (positive_review_ratio + normalized_review_score) / 2
    except (ValueError, TypeError):
        return None

def add_user_reception_score(
    input_path: str,
    output_path: str
) -> None:
    """
    Adds user_reception_score to each game in the dataset.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    for app_id, game in data.items():
        positive_review_ratio = game.get('positive_review_ratio')
        matched_reviews = game.get('matched_reviews', [])
        user_reception_score = calculate_user_reception_score(positive_review_ratio, matched_reviews)
        game['user_reception_score'] = user_reception_score

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main() -> None:
    """
    Main function for CLI usage.
    """
    input_path = "matched_data/games_reviews_concurrent_vscore_posrr.json"
    output_path = "matched_data/games_reviews_concurrent_userrecep.json"
    add_user_reception_score(input_path, output_path)
    print(f"User reception scores added and saved to {output_path}")

if __name__ == "__main__":
    main() 