#!/usr/bin/env python3
"""
Match Steam reviews.csv with games2.json by app_id.

This script reads game data from games2.json and review data from reviews.csv,
then matches them by app_id to create a combined dataset for analysis.
"""

import json
import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, Any, List

def load_games_data(file_path: str) -> Dict[str, Any]:
    """
    Load games data from JSON file.
    
    Args:
        file_path: Path to the games2.json file
        
    Returns:
        Dictionary of game data indexed by app_id
    """
    print(f"Loading games data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            games_data = json.load(f)
        print(f"Successfully loaded data for {len(games_data)} games")
        return games_data
    except Exception as e:
        print(f"Error loading games data: {e}")
        return {}

def load_reviews_data(file_path: str) -> pd.DataFrame:
    """
    Load reviews data from CSV file.
    
    Args:
        file_path: Path to the reviews.csv file
        
    Returns:
        DataFrame containing review data
    """
    print(f"Loading reviews data from {file_path}...")
    try:
        # First attempt - standard parsing
        reviews_df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(reviews_df)} reviews")
        return reviews_df
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        print("Attempting to load with error_bad_lines=False...")
        try:
            # Second attempt - skip bad lines
            reviews_df = pd.read_csv(file_path, on_bad_lines='skip')
            print(f"Successfully loaded {len(reviews_df)} reviews (some rows may have been skipped)")
            return reviews_df
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            try:
                # Third attempt - use Python's csv module for more robust parsing
                print("Attempting manual CSV parsing...")
                import csv
                
                # First, determine the dialect and delimiter
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    sample = f.read(4096)
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    f.seek(0)
                    
                    # Read the header
                    reader = csv.reader(f, dialect)
                    headers = next(reader)
                    
                    # Read all rows
                    rows = []
                    for i, row in enumerate(reader, 1):
                        try:
                            # Handle rows with too many fields
                            if len(row) > len(headers):
                                # Try to combine fields that might have been split incorrectly
                                fixed_row = []
                                j = 0
                                while j < len(row):
                                    if j < len(headers) - 1:
                                        fixed_row.append(row[j])
                                        j += 1
                                    else:
                                        # Combine all remaining fields
                                        fixed_row.append(','.join(row[j:]))
                                        break
                                row = fixed_row
                            
                            # Handle rows with too few fields
                            while len(row) < len(headers):
                                row.append('')
                                
                            rows.append(row)
                        except Exception as row_error:
                            print(f"Error on row {i}: {row_error}")
                            continue
                
                # Create DataFrame
                reviews_df = pd.DataFrame(rows, columns=headers)
                print(f"Successfully loaded {len(reviews_df)} reviews through manual parsing")
                return reviews_df
            except Exception as e3:
                print(f"All parsing attempts failed: {e3}")
                
                # Last resort - try with a specific delimiter and quoting
                try:
                    reviews_df = pd.read_csv(file_path, sep=',', quoting=csv.QUOTE_NONE, escapechar='\\', on_bad_lines='skip')
                    print(f"Successfully loaded {len(reviews_df)} reviews with minimal parsing")
                    return reviews_df
                except Exception as e4:
                    print(f"Final attempt failed: {e4}")
                    return pd.DataFrame()
    except Exception as e:
        print(f"Error loading reviews data: {e}")
        return pd.DataFrame()

def match_by_app_id(games_data: Dict[str, Any], reviews_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Match games with reviews by app_id.
    
    Args:
        games_data: Dictionary of game data indexed by app_id
        reviews_df: DataFrame containing review data
        
    Returns:
        Dictionary of matched data
    """
    print("Matching games with reviews by app_id...")
    
    # Ensure app_id in reviews is treated as string for matching
    if 'app_id' in reviews_df.columns:
        reviews_df['app_id'] = reviews_df['app_id'].astype(str)
    
    matched_data = {}
    matched_count = 0
    games_with_reviews = set()
    
    # Group reviews by app_id for efficient matching
    reviews_grouped = reviews_df.groupby('app_id')
    
    for app_id, game_info in tqdm(games_data.items()):
        try:
            # Check if this game has reviews
            if app_id in reviews_grouped.groups:
                game_reviews = reviews_grouped.get_group(app_id)
                
                # Add reviews to game data
                matched_data[app_id] = {
                    **game_info,  # Original game data
                    'matched_reviews': game_reviews.to_dict('records'),
                    'review_count': len(game_reviews)
                }
                
                matched_count += 1
                games_with_reviews.add(app_id)
            else:
                # Keep the game without reviews
                matched_data[app_id] = {
                    **game_info,
                    'matched_reviews': [],
                    'review_count': 0
                }
        except Exception as e:
            print(f"Error matching game {app_id}: {e}")
    
    print(f"Matched {matched_count} games with reviews out of {len(games_data)} total games")
    print(f"Total games with at least one review: {len(games_with_reviews)}")
    
    return matched_data

def save_matched_data(matched_data: Dict[str, Any], output_file: str) -> None:
    """
    Save matched data to output file.
    
    Args:
        matched_data: Dictionary of matched game and review data
        output_file: Path to save the output file
    """
    print(f"Saving matched data to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, indent=2)
        print(f"Successfully saved matched data to {output_file}")
    except Exception as e:
        print(f"Error saving matched data: {e}")

def create_analysis_file(matched_data: Dict[str, Any], output_file: str) -> None:
    """
    Create a CSV file for analysis with key metrics.
    
    Args:
        matched_data: Dictionary of matched game and review data
        output_file: Path to save the analysis CSV file
    """
    print(f"Creating analysis file {output_file}...")
    
    analysis_data = []
    
    for app_id, game_data in matched_data.items():
        # Extract reviews
        reviews = game_data.get('matched_reviews', [])
        
        # Calculate review metrics if reviews exist
        avg_score = 0
        positive_count = 0
        negative_count = 0
        
        if reviews:
            # Assuming reviews have a 'score' or 'recommended' field
            if 'score' in reviews[0]:
                scores = [r.get('score', 0) for r in reviews]
                avg_score = sum(scores) / len(scores) if scores else 0
            elif 'recommended' in reviews[0]:
                positive_count = sum(1 for r in reviews if r.get('recommended', False))
                negative_count = len(reviews) - positive_count
                avg_score = positive_count / len(reviews) if reviews else 0
        
        # Get language count
        language_count = 0
        if 'supported_languages' in game_data:
            if isinstance(game_data['supported_languages'], list):
                language_count = len(game_data['supported_languages'])
            elif isinstance(game_data['supported_languages'], str):
                language_count = len(game_data['supported_languages'].split(','))
        
        # Create analysis entry
        analysis_entry = {
            'app_id': app_id,
            'name': game_data.get('name', 'Unknown'),
            'language_count': language_count,
            'review_count': len(reviews),
            'avg_score': avg_score,
            'positive_reviews': positive_count,
            'negative_reviews': negative_count,
            'metacritic_score': game_data.get('metacritic_score', 0),
            'recommendations': game_data.get('recommendations', 0),
            'release_date': game_data.get('release_date', '')
        }
        
        analysis_data.append(analysis_entry)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(analysis_data)
    df.to_csv(output_file, index=False)
    print(f"Successfully created analysis file with {len(df)} entries")

def main() -> None:
    """Main function to execute the matching process."""
    # Define input and output files
    games_file = 'games2.json'
    reviews_file = 'reviews.csv'
    output_file = 'games_with_reviews.json'
    analysis_file = 'games_reviews_analysis.csv'
    
    # Create output directory if it doesn't exist
    output_dir = 'matched_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    games_data = load_games_data(games_file)
    reviews_df = load_reviews_data(reviews_file)
    
    if not games_data or reviews_df.empty:
        print("Cannot proceed without both games and reviews data.")
        return
    
    # Match data
    matched_data = match_by_app_id(games_data, reviews_df)
    
    # Save results
    save_matched_data(matched_data, os.path.join(output_dir, output_file))
    
    # Create analysis file
    create_analysis_file(matched_data, os.path.join(output_dir, analysis_file))
    
    print("Matching process complete!")

if __name__ == "__main__":
    main() 