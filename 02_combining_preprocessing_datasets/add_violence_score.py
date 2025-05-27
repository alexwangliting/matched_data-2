import json
from typing import Dict, Any

VIOLENCE_TAGS = [
    'Shooter', 'Horror', 'Combat', 'War', 'Violent', 'FPS',
    'Psychological Horror', 'Gore'
]


def compute_violence_score(tags: Dict[str, int]) -> int:
    """
    Compute the violence score for a game based on its tags.

    Args:
        tags: Dictionary of tag names to their counts.

    Returns:
        The number of violence-related tags present.
    """
    violence_tags_lower = {tag.lower() for tag in VIOLENCE_TAGS}
    return sum(
        1 for tag in tags
        if tag.lower() in violence_tags_lower
    )


def add_violence_score_to_games(
    input_path: str,
    output_path: str
) -> None:
    """
    Adds a violence score to each game in the dataset.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    for app_id, game in data.items():
        tags = game.get('tags', {})
        violence_score = compute_violence_score(tags)
        game['violence_score'] = violence_score

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    """
    Main function for CLI usage.
    """
    input_path = "matched_data/games_with_reviews_and_concurrent.json"
    output_path = "matched_data/games_reviews_concurrent_vscore.json"
    add_violence_score_to_games(input_path, output_path)
    print(f"Violence scores added and saved to {output_path}")


if __name__ == "__main__":
    main() 