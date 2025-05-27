import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

def count_violence_tags(filepath: str, violence_tags: List[str]) -> Dict[str, int]:
    """Count the number of games with each violence-related tag.

    Args:
        filepath (str): Path to the JSON file.
        violence_tags (List[str]): List of violence-related tags to count.

    Returns:
        Dict[str, int]: Mapping from tag to count.
    """
    tag_counts = {tag: 0 for tag in violence_tags}
    with open(filepath, 'r') as f:
        data = json.load(f)
        for game in data.values():
            game_tags = game.get('tags', {})
            tags_set = set()
            if isinstance(game_tags, dict):
                tags_set = set(game_tags.keys())
            elif isinstance(game_tags, list):
                tags_set = set(game_tags)
            for tag in violence_tags:
                if tag in tags_set:
                    tag_counts[tag] += 1
    return tag_counts, len(data)

def plot_violence_tags_bar(tags: List[str], percentages: List[float], output_dir: str) -> None:
    """Plot and save a bar chart for violence-related tags.

    Args:
        tags (List[str]): List of tag names.
        percentages (List[float]): Corresponding percentages.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=percentages, y=tags, orient='h', palette='Reds_r')
    plt.xlabel('Percentage (%)')
    plt.ylabel('Tag')
    plt.title('Top 8 Violence-Related Tags')
    plt.xlim(0, max(percentages) + 2)
    for i, v in enumerate(percentages):
        plt.text(v + 0.2, i, f'{v:.2f}%', va='center')
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'top_violence_tags_barchart.png')
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

def main() -> None:
    """Main function to plot the top 8 violence-related tags as a bar chart."""
    input_path = 'matched_data/full_data.json'
    output_dir = 'matched_data/tags_barchart'
    violence_tags = [
        'Shooter', 'Horror', 'Combat', 'War', 'Violent',
        'FPS', 'Psychological Horror', 'Gore'
    ]
    tag_counts, total_games = count_violence_tags(input_path, violence_tags)
    percentages = [100 * tag_counts[tag] / total_games for tag in violence_tags]
    plot_violence_tags_bar(violence_tags, percentages, output_dir)

if __name__ == '__main__':
    main() 