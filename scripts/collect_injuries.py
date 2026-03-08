"""
Collect NBA injury report data using the nbainjuries package.
Data available from 2021-22 season onward (NBA official server snapshots).

Usage:
    python scripts/collect_injuries.py                    # Collect current season
    python scripts/collect_injuries.py --season 2024      # Collect specific season
    python scripts/collect_injuries.py --all               # Collect all available (2022-2026)

Output: data/injuries.csv
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import pandas as pd
from nbainjuries import injury

DATA_DIR = Path(__file__).parent.parent / "data"

# NBA regular season approximate date ranges
SEASON_DATES = {
    2022: (datetime.date(2021, 10, 19), datetime.date(2022, 4, 10)),
    2023: (datetime.date(2022, 10, 18), datetime.date(2023, 4, 9)),
    2024: (datetime.date(2023, 10, 24), datetime.date(2024, 4, 14)),
    2025: (datetime.date(2024, 10, 22), datetime.date(2025, 4, 13)),
    2026: (datetime.date(2025, 10, 28), datetime.date(2026, 4, 12)),
}


def collect_season(season: int) -> pd.DataFrame:
    """Collect injury reports for an entire season, one snapshot per game day at 5pm ET."""
    if season not in SEASON_DATES:
        print(f"  Season {season} not in range (2022-2026)")
        return pd.DataFrame()

    start, end = SEASON_DATES[season]
    today = datetime.date.today()
    if end > today:
        end = today - datetime.timedelta(days=1)

    print(f"  Collecting {season}: {start} to {end}")

    all_reports = []
    current = start
    skipped = 0
    collected = 0

    while current <= end:
        timestamp = datetime.datetime(current.year, current.month, current.day, 17, 0)

        try:
            valid = injury.check_reportvalid(timestamp)
            if valid:
                df = injury.get_reportdata(timestamp, return_df=True)
                if df is not None and len(df) > 0:
                    df["report_date"] = current.isoformat()
                    df["season"] = season
                    all_reports.append(df)
                    collected += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1

        current += datetime.timedelta(days=1)
        time.sleep(0.2)

        if collected % 20 == 0 and collected > 0:
            print(f"    ... {collected} reports collected, {skipped} skipped")

    print(f"  Season {season}: {collected} reports, {skipped} skipped")

    if all_reports:
        return pd.concat(all_reports, ignore_index=True)
    return pd.DataFrame()


def normalize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize injury data for feature engineering."""
    if df.empty:
        return df

    df = df.rename(columns={
        "Game Date": "game_date",
        "Game Time": "game_time",
        "Matchup": "matchup",
        "Team": "team_full",
        "Player Name": "player",
        "Current Status": "status",
        "Reason": "reason",
    })

    df["is_out"] = df["status"].str.contains("Out", case=False, na=False).astype(int)
    df["is_questionable"] = df["status"].str.contains("Questionable", case=False, na=False).astype(int)
    df["is_doubtful"] = df["status"].str.contains("Doubtful", case=False, na=False).astype(int)
    df["is_probable"] = df["status"].str.contains("Probable|Available", case=False, na=False).astype(int)

    df["is_injury"] = df["reason"].str.contains("Injury|Illness", case=False, na=False).astype(int)
    df["is_rest"] = df["reason"].str.contains("Rest", case=False, na=False).astype(int)
    df["is_gleague"] = df["reason"].str.contains("G League|G-League", case=False, na=False).astype(int)

    def normalize_name(name):
        if pd.isna(name):
            return name
        if "," in str(name):
            parts = str(name).split(",", 1)
            return f"{parts[1].strip()} {parts[0].strip()}"
        return str(name).strip()

    df["player"] = df["player"].apply(normalize_name)

    return df


def main():
    parser = argparse.ArgumentParser(description="Collect NBA injury data")
    parser.add_argument("--season", type=int, help="Specific season to collect")
    parser.add_argument("--all", action="store_true", help="Collect all available seasons")
    args = parser.parse_args()

    if args.all:
        seasons = sorted(SEASON_DATES.keys())
    elif args.season:
        seasons = [args.season]
    else:
        seasons = [2026]

    print(f"Collecting injury data for seasons: {seasons}")

    all_data = []
    for season in seasons:
        df = collect_season(season)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        print("No data collected!")
        sys.exit(1)

    combined = pd.concat(all_data, ignore_index=True)
    combined = normalize_injuries(combined)

    out_path = DATA_DIR / "injuries.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} injury records to {out_path}")
    print(f"  Seasons: {sorted(combined['season'].unique())}")
    print(f"  Date range: {combined['report_date'].min()} to {combined['report_date'].max()}")
    print(f"  Unique players: {combined['player'].nunique()}")


if __name__ == "__main__":
    main()
