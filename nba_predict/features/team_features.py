"""
Prior-season team quality features.

Joins teams_advanced.csv and teams_per_game.csv from season N-1
onto games in season N. This is safe because an entire prior season
is complete before the new season starts.
"""

import pandas as pd

from nba_predict.config import DATA_DIR
from nba_predict.data.loader import load_teams_advanced, load_teams_per_game, load_teams_opponent
from nba_predict.data.cleaning import safe_float


# Columns to extract from teams_advanced for prior-season features
ADVANCED_COLS = [
    "off_rtg", "def_rtg", "net_rtg", "pace", "ts_pct", "srs", "sos", "mov",
    # Four Factors (offense + defense) — proven predictors of team quality
    "efg_pct", "tov_pct", "orb_pct", "ft_rate",
    "opp_efg_pct", "opp_tov_pct", "drb_pct", "opp_ft_rate",
]

# Columns from teams_per_game
PER_GAME_COLS = ["fg_pct", "fg3_pct", "ft_pct", "trb", "ast", "stl", "blk", "tov", "pts"]


def build_prior_season_features() -> pd.DataFrame:
    """Build a lookup table of prior-season team stats.

    Returns DataFrame with columns: team, season, {stat}_prev for each stat.
    The 'season' column is the PREDICTION season (N), and the stats come from N-1.
    """
    # Load team stats
    adv = load_teams_advanced()
    tpg = load_teams_per_game()
    opp = load_teams_opponent()

    # Extract relevant columns from each source
    adv_subset = adv[["team", "season"] + [c for c in ADVANCED_COLS if c in adv.columns]].copy()
    tpg_subset = tpg[["team", "season"] + [c for c in PER_GAME_COLS if c in tpg.columns]].copy()

    # For opponent stats, prefix with opp_
    opp_cols_to_use = [c for c in PER_GAME_COLS if c in opp.columns]
    opp_subset = opp[["team", "season"] + opp_cols_to_use].copy()
    opp_subset = opp_subset.rename(columns={c: f"opp_{c}" for c in opp_cols_to_use})

    # Merge all team stats for the same season
    merged = adv_subset.merge(tpg_subset, on=["team", "season"], how="outer", suffixes=("", "_pg"))
    merged = merged.merge(opp_subset, on=["team", "season"], how="outer")

    # Shift season forward: stats from season N become features for season N+1
    merged["season"] = merged["season"] + 1

    # Rename all stat columns with _prev suffix
    rename_map = {}
    for col in merged.columns:
        if col not in ("team", "season"):
            rename_map[col] = f"{col}_prev"
    merged = merged.rename(columns=rename_map)

    return merged


def join_prior_season(games: pd.DataFrame, prior_stats: pd.DataFrame,
                      team_col: str, prefix: str) -> pd.DataFrame:
    """Join prior-season stats onto a games DataFrame for a specific team column.

    Args:
        games: games DataFrame with team_col and 'season' columns
        prior_stats: output of build_prior_season_features()
        team_col: column name in games to join on (e.g., 'team_home', 'team_away')
        prefix: prefix for the joined columns (e.g., 'home_', 'away_')
    """
    # Rename prior_stats columns with prefix
    stats_renamed = prior_stats.copy()
    rename_map = {col: f"{prefix}{col}" for col in stats_renamed.columns
                  if col not in ("team", "season")}
    stats_renamed = stats_renamed.rename(columns=rename_map)

    result = games.merge(
        stats_renamed,
        left_on=[team_col, "season"],
        right_on=["team", "season"],
        how="left",
    )

    # Drop the duplicate 'team' column from the join
    if "team_x" in result.columns:
        result = result.rename(columns={"team_x": team_col}).drop(columns=["team_y"], errors="ignore")
    elif "team" in result.columns and team_col != "team":
        result = result.drop(columns=["team"], errors="ignore")

    return result
