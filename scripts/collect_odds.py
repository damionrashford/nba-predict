"""
Collect NBA betting lines from The Odds API.

Free tier: 500 requests/month, live odds only (no historical).
Historical odds require paid plan ($12-79/mo).

Usage:
    python scripts/collect_odds.py                        # Collect today's odds
    python scripts/collect_odds.py --historical 2025      # Historical (paid plan)

Output: data/odds.csv (appends new data, deduplicates)

Set ODDS_API_KEY environment variable or create .env file.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).parent.parent / "data"

# The Odds API endpoints
BASE_URL = "https://api.the-odds-api.com"
SPORT = "basketball_nba"

# Team name mapping: Odds API full names -> our 3-letter codes
ODDS_TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


def get_api_key() -> str:
    """Get API key from environment or .env file."""
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: Set ODDS_API_KEY environment variable or add to .env file")
    print("  Get a free key at: https://the-odds-api.com/")
    sys.exit(1)


def collect_live_odds(api_key: str) -> pd.DataFrame:
    """Collect current live odds (spreads + totals + moneyline)."""
    print("Collecting live NBA odds...")

    url = f"{BASE_URL}/v4/sports/{SPORT}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads,totals,h2h",
        "oddsFormat": "american",
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        return pd.DataFrame()

    remaining = resp.headers.get("x-requests-remaining", "?")
    print(f"  API requests remaining: {remaining}")

    games = resp.json()
    rows = []
    for game in games:
        game_id = game.get("id", "")
        commence = game.get("commence_time", "")
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        home_code = ODDS_TEAM_MAP.get(home, home[:3].upper())
        away_code = ODDS_TEAM_MAP.get(away, away[:3].upper())

        for book in game.get("bookmakers", []):
            book_key = book.get("key", "")
            for market in book.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "game_id": game_id,
                        "commence_time": commence,
                        "home_team": home_code,
                        "away_team": away_code,
                        "bookmaker": book_key,
                        "market": market_key,
                        "outcome_name": outcome.get("name", ""),
                        "price": outcome.get("price"),
                        "point": outcome.get("point"),
                        "collected_at": datetime.utcnow().isoformat(),
                    })

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} odds lines for {len(games)} games")
    return df


def collect_historical_odds(api_key: str, season: int) -> pd.DataFrame:
    """Collect historical odds for a season (requires paid plan)."""
    print(f"Collecting historical odds for season {season}...")
    print("  NOTE: Historical odds require a paid Odds API plan")

    # Season date range
    season_ranges = {
        2024: ("2023-10-24", "2024-04-14"),
        2025: ("2024-10-22", "2025-04-13"),
        2026: ("2025-10-28", "2026-03-06"),
    }

    if season not in season_ranges:
        print(f"  Season {season} not configured")
        return pd.DataFrame()

    start_str, end_str = season_ranges[season]
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    today = datetime.utcnow()
    if end > today:
        end = today - timedelta(days=1)

    url = f"{BASE_URL}/v4/sports/{SPORT}/odds-history/"
    all_rows = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%dT12:00:00Z")
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "spreads,totals,h2h",
            "oddsFormat": "american",
            "date": date_str,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 422:
                print(f"    {current.date()}: No games")
            elif resp.status_code == 401:
                print("    ERROR: Historical odds require paid plan")
                break
            elif resp.status_code == 200:
                data = resp.json()
                games = data.get("data", [])
                for game in games:
                    for book in game.get("bookmakers", []):
                        for market in book.get("markets", []):
                            for outcome in market.get("outcomes", []):
                                all_rows.append({
                                    "game_id": game.get("id", ""),
                                    "commence_time": game.get("commence_time", ""),
                                    "home_team": ODDS_TEAM_MAP.get(game.get("home_team", ""), ""),
                                    "away_team": ODDS_TEAM_MAP.get(game.get("away_team", ""), ""),
                                    "bookmaker": book.get("key", ""),
                                    "market": market.get("key", ""),
                                    "outcome_name": outcome.get("name", ""),
                                    "price": outcome.get("price"),
                                    "point": outcome.get("point"),
                                    "collected_at": date_str,
                                    "season": season,
                                })
                remaining = resp.headers.get("x-requests-remaining", "?")
                print(f"    {current.date()}: {len(games)} games (remaining: {remaining})")
            else:
                print(f"    {current.date()}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"    {current.date()}: ERROR {e}")

        current += timedelta(days=1)

    return pd.DataFrame(all_rows)


def pivot_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw odds into one row per game with consensus lines."""
    if df.empty:
        return df

    games = []
    for (gid, home, away, commence), gdf in df.groupby(
        ["game_id", "home_team", "away_team", "commence_time"]
    ):
        row = {
            "game_id": gid,
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
        }

        # Consensus spread (median across books)
        spreads = gdf[(gdf["market"] == "spreads") & (gdf["outcome_name"].str.contains(str(home), case=False, na=False))]
        if not spreads.empty and "point" in spreads.columns:
            row["spread_home"] = spreads["point"].median()

        # Consensus total
        totals = gdf[(gdf["market"] == "totals") & (gdf["outcome_name"].str.lower() == "over")]
        if not totals.empty and "point" in totals.columns:
            row["total"] = totals["point"].median()

        # Moneyline
        ml_home = gdf[(gdf["market"] == "h2h") & (gdf["outcome_name"].str.contains(str(home), case=False, na=False))]
        ml_away = gdf[(gdf["market"] == "h2h") & (gdf["outcome_name"].str.contains(str(away), case=False, na=False))]
        if not ml_home.empty:
            row["ml_home"] = ml_home["price"].median()
        if not ml_away.empty:
            row["ml_away"] = ml_away["price"].median()

        games.append(row)

    return pd.DataFrame(games)


def main():
    parser = argparse.ArgumentParser(description="Collect NBA betting odds")
    parser.add_argument("--historical", type=int, help="Collect historical season (paid plan)")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.historical:
        raw = collect_historical_odds(api_key, args.historical)
    else:
        raw = collect_live_odds(api_key)

    if raw.empty:
        print("No odds data collected!")
        sys.exit(1)

    # Save raw
    raw_path = DATA_DIR / "odds_raw.csv"
    if raw_path.exists():
        existing = pd.read_csv(raw_path)
        raw = pd.concat([existing, raw], ignore_index=True)
        raw = raw.drop_duplicates(subset=["game_id", "bookmaker", "market", "outcome_name", "collected_at"])
    raw.to_csv(raw_path, index=False)
    print(f"\nSaved {len(raw)} raw odds lines to {raw_path}")

    # Pivot to game-level consensus
    pivoted = pivot_odds(raw)
    out = DATA_DIR / "odds.csv"
    if out.exists():
        existing = pd.read_csv(out)
        pivoted = pd.concat([existing, pivoted], ignore_index=True)
        pivoted = pivoted.drop_duplicates(subset=["game_id"])
    pivoted.to_csv(out, index=False)
    print(f"Saved {len(pivoted)} game odds to {out}")


if __name__ == "__main__":
    main()
