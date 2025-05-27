import json
from typing import Dict, Any

def add_no_sup_lang(
    input_path: str,
    output_path: str
) -> None:
    """
    Adds the number of supported languages (no_sup_lang) to each game in the dataset.

    Args:
        input_path: Path to the input JSON file.
        output_path: Path to the output JSON file.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: Dict[str, Any] = json.load(f)

    for app_id, game in data.items():
        supported_languages = game.get('supported_languages', [])
        if isinstance(supported_languages, list):
            no_sup_lang = len(supported_languages)
        else:
            no_sup_lang = 0
        game['no_sup_lang'] = no_sup_lang

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main() -> None:
    """
    Main function for CLI usage.
    """
    input_path = "matched_data/games_reviews_concurrent_userrecep.json"
    output_path = "matched_data/full_data.json"
    add_no_sup_lang(input_path, output_path)
    print(f"Number of supported languages added and saved to {output_path}")

if __name__ == "__main__":
    main() 