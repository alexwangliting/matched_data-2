#!/usr/bin/env python3
"""
Combine concurrent_users_yesterday data from steamspy_insights.csv with games_with_reviews.json.

This script extracts the concurrent_users_yesterday column from the steamspy_insights.csv file
and adds it to the games_with_reviews.json file, matching by app_id.
"""

import json
import pandas as pd
from typing import Dict, Any
import os

def load_steamspy_data(steamspy_file: str) -> pd.DataFrame:
    """
    Load concurrent users data from steamspy_insights.csv.
    
    Args:
        steamspy_file: Path to the steamspy_insights.csv file
        
    Returns:
        DataFrame with app_id and concurrent_users_yesterday columns
    """
    print(f"Loading SteamSpy data from {steamspy_file}...")
    
    # Read only the needed columns to save memory
    # Use string type for all columns initially to avoid parsing errors
    steamspy_df = pd.read_csv(
        steamspy_file, 
        usecols=["app_id", "concurrent_users_yesterday"],
        dtype={"app_id": str, "concurrent_users_yesterday": str}
    )
    
    # Ensure app_id is treated as string
    steamspy_df['app_id'] = steamspy_df['app_id'].astype(str)
    
    # Convert concurrent_users_yesterday to numeric, errors='coerce' will set invalid values to NaN
    steamspy_df['concurrent_users_yesterday'] = pd.to_numeric(
        steamspy_df['concurrent_users_yesterday'], 
        errors='coerce'
    )
    
    print(f"Loaded {len(steamspy_df)} games with concurrent users data")
    print(f"Games with valid concurrent users data: {steamspy_df['concurrent_users_yesterday'].notna().sum()}")
    
    return steamspy_df

def load_reviews_data(reviews_file: str) -> Dict[str, Any]:
    """
    Load existing games_with_reviews.json file.
    
    Args:
        reviews_file: Path to the games_with_reviews.json file
        
    Returns:
        Dictionary containing game data
    """
    print(f"Loading review data from {reviews_file}...")
    with open(reviews_file, 'r', encoding='utf-8') as f:
        review_data = json.load(f)
    
    print(f"Loaded {len(review_data)} games with review data")
    return review_data

def combine_data(steamspy_df: pd.DataFrame, review_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine the concurrent users data with review data.
    
    Args:
        steamspy_df: DataFrame with app_id and concurrent_users_yesterday
        review_data: Dictionary of game data from games_with_reviews.json
        
    Returns:
        Updated dictionary with concurrent users data added
    """
    print("Combining concurrent users data with review data...")
    
    # Create a dictionary for faster lookups
    concurrent_users_dict = dict(zip(
        steamspy_df['app_id'],
        steamspy_df['concurrent_users_yesterday']
    ))
    
    # Count for stats
    games_updated = 0
    
    # Add concurrent_users_yesterday to each game in review_data
    for app_id, game_data in review_data.items():
        if app_id in concurrent_users_dict:
            # Get the concurrent users value
            concurrent_users = concurrent_users_dict[app_id]
            
            # Convert NaN to None for JSON compatibility
            if pd.isna(concurrent_users):
                game_data['concurrent_users_yesterday'] = None
            else:
                game_data['concurrent_users_yesterday'] = int(concurrent_users)
            
            games_updated += 1
    
    print(f"Added concurrent users data to {games_updated} games")
    return review_data

def save_updated_data(data: Dict[str, Any], output_file: str) -> None:
    """
    Save the updated data to a JSON file.
    
    Args:
        data: Updated game data dictionary
        output_file: Path to save the updated JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving updated data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data successfully saved!")

def main() -> None:
    """Main function to execute the data combination."""
    steamspy_file = '/Users/wang/y4s2/3_THURam_bigdata/steam-insights-main/steamspy_insights.csv'
    reviews_file = 'matched_data/games_with_reviews.json'
    output_file = 'matched_data/games_with_reviews_and_concurrent.json'
    
    print("Starting combination of concurrent users data with game reviews...")
    
    # Load data
    steamspy_df = load_steamspy_data(steamspy_file)
    review_data = load_reviews_data(reviews_file)
    
    # Combine data
    updated_data = combine_data(steamspy_df, review_data)
    
    # Save updated data
    save_updated_data(updated_data, output_file)
    
    print("Process completed successfully!")
    print(f"Original games count: {len(review_data)}")
    print(f"Updated games with concurrent users data saved to: {output_file}")

if __name__ == "__main__":
    main() 