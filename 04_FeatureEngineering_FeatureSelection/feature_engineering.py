import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List
import re

INPUT_PATH = "matched_data/full_data.json"
CSV_OUT = "matched_data/engineered_features.csv"
JSON_OUT = "matched_data/engineered_features.json"

TODAY = datetime.now().date()


def parse_estimated_owners(owners: str) -> float:
    """Convert estimated owners range string to midpoint float."""
    if not owners or '-' not in owners:
        return np.nan
    parts = re.findall(r"\d+", owners)
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return np.nan


def parse_date(date_str: str) -> int:
    """Return game age in days from release_date to today."""
    try:
        dt = datetime.strptime(date_str, "%b %d, %Y").date()
        return (TODAY - dt).days
    except Exception:
        return np.nan


def main() -> None:
    """Extract and engineer features for modeling from full_data.json."""
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for game_id, g in data.items():
        row = {
            "game_id": game_id,
            "game_age_days": parse_date(g.get("release_date", "")),
            "dlc_count": g.get("dlc_count", 0),
            "achievements": g.get("achievements", 0),
            "is_multiplayer": int("Multi-player" in g.get("categories", [])),
            "n_categories": len(g.get("categories", [])),
            "n_genres": len(g.get("genres", [])),
            "n_tags": len(g.get("tags", {})),
            "review_count": g.get("review_count", 0),
            "metacritic_score": int(g.get("metacritic_score", 0) or 0),
            "estimated_owners": parse_estimated_owners(g.get("estimated_owners", "")),
            "windows": int(g.get("windows", False)),
            "mac": int(g.get("mac", False)),
            "linux": int(g.get("linux", False)),
            "no_sup_lang": len(g.get("supported_languages", [])),
            "n_full_audio_lang": len(g.get("full_audio_languages", [])),
            "price": g.get("price", 0),
            "developer": ",".join(g.get("developers", [])),
            "publisher": ",".join(g.get("publishers", [])),
            "user_reception_score": g.get("user_reception_score", np.nan),
            "log_concurrent_users_yesterday": np.log1p(g.get("concurrent_users_yesterday") or 0),
            "violence_score": g.get("violence_score", 0),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    df.to_json(JSON_OUT, orient="records")
    print(f"Saved {CSV_OUT} and {JSON_OUT}")

if __name__ == "__main__":
    main() 