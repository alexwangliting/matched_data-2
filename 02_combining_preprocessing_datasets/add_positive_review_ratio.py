import json
from typing import Dict, Any, Optional

def calculate_positive_review_ratio(matched_reviews: list[dict[str, Any]]) -> Optional[float]:
    """
    Calculate the positive review ratio from matched_reviews.

    Args:
        matched_reviews: List of review dicts, each with 'positive' and 'total' fields as strings.

    Returns:
        The positive review ratio (positive/total) as a float, or None if not computable.
    """
    if not matched_reviews:
        return None
    review = matched_reviews[0]
    try:
        positive = int(review.get('positive', 0))
        total = int(review.get('total', 0))
        if total > 0:
            return positive / total
        return None
    except (ValueError, TypeError):
        return None

def add_positive_review_ratio(
    input_path: str,
    output_path: str
) -> None:
    """
    Adds positive review ratio to each game in the dataset.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    for app_id, game in data.items():
        matched_reviews = game.get('matched_reviews', [])
        ratio = calculate_positive_review_ratio(matched_reviews)
        game['positive_review_ratio'] = ratio

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main() -> None:
    """
    Main function for CLI usage.
    """
    input_path = "matched_data/games_reviews_concurrent_vscore.json"
    output_path = "matched_data/games_reviews_concurrent_vscore_posrr.json"
    add_positive_review_ratio(input_path, output_path)
    print(f"Positive review ratios added and saved to {output_path}")

if __name__ == "__main__":
    main() 