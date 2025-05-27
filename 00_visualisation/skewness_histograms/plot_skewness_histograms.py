import json
import os
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_json_data(filepath: str, variables: List[str]) -> dict:
    """Load selected variables from a large JSON file.

    Args:
        filepath (str): Path to the JSON file.
        variables (List[str]): List of variable names to extract.

    Returns:
        dict: Dictionary with variable names as keys and lists of values.
    """
    data = {var: [] for var in variables}
    with open(filepath, 'r') as f:
        raw = json.load(f)
        for game in raw.values():
            for var in variables:
                val = game.get(var, None)
                if val is not None:
                    data[var].append(val)
    return data


def plot_histogram(data: List[float], var_name: str, log: bool = False, output_dir: str = '.') -> None:
    """Plot and save a histogram for a variable.

    Args:
        data (List[float]): Data to plot.
        var_name (str): Variable name for labeling.
        log (bool): Whether to plot log-transformed data.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    if log:
        data = [x for x in data if x > 0]
        data = np.log10(data)
        sns.histplot(data, bins=50, kde=True, color='orange')
        plt.xlabel(f'log10({var_name})')
        plt.title(f'Histogram of log10({var_name})')
        filename = f'{output_dir}/hist_{var_name}_log10.png'
    else:
        sns.histplot(data, bins=50, kde=True, color='blue')
        plt.xlabel(var_name)
        plt.title(f'Histogram of {var_name}')
        filename = f'{output_dir}/hist_{var_name}.png'
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def main() -> None:
    """Main function to plot histograms for price and concurrent_users_yesterday."""
    input_path = 'matched_data/full_data.json'
    output_dir = 'matched_data/skewness_histograms'
    os.makedirs(output_dir, exist_ok=True)
    variables = ['price', 'concurrent_users_yesterday', 'user_reception_score', 'no_sup_lang', 'violence_score']
    data = load_json_data(input_path, variables)
    for var in variables:
        plot_histogram(data[var], var, log=False, output_dir=output_dir)
        if any(x > 0 for x in data[var]):
            plot_histogram(data[var], var, log=True, output_dir=output_dir)

if __name__ == '__main__':
    main() 