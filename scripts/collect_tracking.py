"""
Collect NBA player tracking + clutch stats from nba_api (stats.nba.com).
Data available from 2013-14 season onward.

Collects:
  1. Player tracking stats (distance, speed, touches) via LeagueDashPtStats
  2. Player clutch stats (last 5 min, <=5 pt game) via LeagueDashPlayerClutch

Usage:
    python scripts/collect_tracking.py                    # Current season
    python scripts/collect_tracking.py --season 2025      # Specific season
    python scripts/collect_tracking.py --all              # All available (2014-2026)

Output: data/tracking.csv, data/clutch.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# Tracking data available from 2013-14 onward
TRACKING_FIRST_SEASON = 2014
CURRENT_SEASON = 2026


def season_str(season_end_year: int) -> str:
    """Convert season end year to NBA API format: 2025 -> '2024-25'."""
    start = season_end_year - 1
    return f"{start}-{str(season_end_year)[-2:]}"


def collect_tracking_season(season: int) -> pd.DataFrame:
    """Collect player tracking stats for one season."""
    from nba_api.stats.endpoints import LeagueDashPtStats

    ss = season_str(season)
    print(f"  Tracking {ss}...")

    try:
        stats = LeagueDashPtStats(
            season=ss,
            per_mode_simple="PerGame",
            player_or_team="Player",
            timeout=60,
        )
        df = stats.get_data_frames()[0]
        df["season"] = season
        print(f"    -> {len(df)} players")
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def collect_clutch_season(season: int) -> pd.DataFrame:
    """Collect player clutch stats for one season."""
    from nba_api.stats.endpoints import LeagueDashPlayerClutch

    ss = season_str(season)
    print(f"  Clutch {ss}...")

    try:
        stats = LeagueDashPlayerClutch(
            season=ss,
            per_mode_detailed="PerGame",
            clutch_time="Last 5 Minutes",
            ahead_behind="Ahead or Behind",
            point_diff=5,
            timeout=60,
        )
        df = stats.get_data_frames()[0]
        df["season"] = season
        print(f"    -> {len(df)} players")
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return pd.DataFrame()


def normalize_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename useful tracking columns."""
    if df.empty:
        return df

    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "season",
        "GP", "MIN", "DIST_MILES", "DIST_MILES_OFF", "DIST_MILES_DEF",
        "AVG_SPEED", "AVG_SPEED_OFF", "AVG_SPEED_DEF",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    rename = {
        "PLAYER_NAME": "player",
        "TEAM_ABBREVIATION": "team",
        "GP": "tracking_gp",
        "MIN": "tracking_min",
        "DIST_MILES": "dist_miles",
        "DIST_MILES_OFF": "dist_miles_off",
        "DIST_MILES_DEF": "dist_miles_def",
        "AVG_SPEED": "avg_speed",
        "AVG_SPEED_OFF": "avg_speed_off",
        "AVG_SPEED_DEF": "avg_speed_def",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def normalize_clutch(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename useful clutch columns."""
    if df.empty:
        return df

    keep = [
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "season",
        "GP", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "REB", "AST", "TOV", "STL", "BLK",
        "PTS", "PLUS_MINUS",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    rename = {
        "PLAYER_NAME": "player",
        "TEAM_ABBREVIATION": "team",
        "GP": "clutch_gp",
        "MIN": "clutch_min",
        "FG_PCT": "clutch_fg_pct",
        "FG3_PCT": "clutch_fg3_pct",
        "FT_PCT": "clutch_ft_pct",
        "PTS": "clutch_pts",
        "AST": "clutch_ast",
        "REB": "clutch_reb",
        "TOV": "clutch_tov",
        "STL": "clutch_stl",
        "BLK": "clutch_blk",
        "PLUS_MINUS": "clutch_plus_minus",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect NBA tracking + clutch data")
    parser.add_argument("--season", type=int, help="Specific season to collect")
    parser.add_argument("--all", action="store_true", help="Collect all available seasons (2014-2026)")
    args = parser.parse_args()

    if args.all:
        seasons = list(range(TRACKING_FIRST_SEASON, CURRENT_SEASON + 1))
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [CURRENT_SEASON]

    print(f"Collecting tracking + clutch data for seasons: {seasons}")

    # Collect tracking
    tracking_dfs = []
    for s in seasons:
        df = collect_tracking_season(s)
        if not df.empty:
            tracking_dfs.append(df)
        time.sleep(1.5)  # Rate limit nba_api

    if tracking_dfs:
        tracking = pd.concat(tracking_dfs, ignore_index=True)
        tracking = normalize_tracking(tracking)
        out = DATA_DIR / "tracking.csv"
        tracking.to_csv(out, index=False)
        print(f"\nSaved {len(tracking)} tracking records to {out}")
    else:
        print("\nNo tracking data collected!")

    # Collect clutch
    clutch_dfs = []
    for s in seasons:
        df = collect_clutch_season(s)
        if not df.empty:
            clutch_dfs.append(df)
        time.sleep(1.5)

    if clutch_dfs:
        clutch = pd.concat(clutch_dfs, ignore_index=True)
        clutch = normalize_clutch(clutch)
        out = DATA_DIR / "clutch.csv"
        clutch.to_csv(out, index=False)
        print(f"Saved {len(clutch)} clutch records to {out}")
    else:
        print("\nNo clutch data collected!")


if __name__ == "__main__":
    main()
