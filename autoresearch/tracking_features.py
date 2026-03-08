"""
Tracking data features for season-level and game-level predictions.

Uses NBA tracking data (player speed, distance covered) aggregated to
team-level prior-season features. Available for seasons 2014-2026.

Historical seasons before 2014 will have NaN, handled natively by XGBoost.
"""

import pandas as pd

from nba_predict.config import DATA_DIR
from nba_predict.data.team_mapping import normalize_team_name


def load_tracking() -> pd.DataFrame:
    """Load and normalize tracking.csv."""
    df = pd.read_csv(DATA_DIR / "tracking.csv", low_memory=False)

    def _safe_norm(val):
        try:
            return normalize_team_name(str(val).strip())
        except ValueError:
            return None

    df["team"] = df["team"].apply(_safe_norm)
    df = df.dropna(subset=["team", "season"])
    return df


def build_tracking_features() -> pd.DataFrame:
    """Build prior-season team-level tracking features.

    Aggregates player tracking data into team averages, then shifts
    forward by 1 season (season N stats become features for season N+1).

    Returns DataFrame with: team, season (prediction season), tracking_* columns.
    """
    tracking = load_tracking()

    # Filter to meaningful playing time (tracking_min is per-game avg, not total)
    valid = tracking[tracking["tracking_min"] >= 15].copy()

    # Aggregate to team-season level
    agg = valid.groupby(["team", "season"]).agg(
        tracking_avg_speed=("avg_speed", "mean"),
        tracking_avg_speed_off=("avg_speed_off", "mean"),
        tracking_avg_speed_def=("avg_speed_def", "mean"),
        tracking_avg_dist=("dist_miles", "mean"),
        tracking_dist_off=("dist_miles_off", "mean"),
        tracking_dist_def=("dist_miles_def", "mean"),
        tracking_player_count=("player", "count"),
    ).reset_index()

    # Derived: offensive vs defensive effort ratio
    agg["tracking_off_def_speed_ratio"] = (
        agg["tracking_avg_speed_off"] / agg["tracking_avg_speed_def"].replace(0, 1)
    )

    # Shift forward: season N stats become features for season N+1
    agg["season"] = agg["season"] + 1

    return agg


def join_tracking_features(matchups: pd.DataFrame,
                            tracking_feats: pd.DataFrame) -> pd.DataFrame:
    """Join prior-season tracking features onto matchup dataset."""
    track = tracking_feats.copy()

    # Join for home team
    home_track = track.copy()
    home_rename = {c: f"home_{c}" for c in home_track.columns if c not in ("team", "season")}
    home_track = home_track.rename(columns=home_rename)
    matchups = matchups.merge(
        home_track, left_on=["team_home", "season"], right_on=["team", "season"], how="left",
    )
    matchups = matchups.drop(columns=["team"], errors="ignore")

    # Join for away team
    away_track = track.copy()
    away_rename = {c: f"away_{c}" for c in away_track.columns if c not in ("team", "season")}
    away_track = away_track.rename(columns=away_rename)
    matchups = matchups.merge(
        away_track, left_on=["team_away", "date"], right_on=["team", "date"], how="left",
    ) if "date" in away_track.columns else matchups.merge(
        away_track, left_on=["team_away", "season"], right_on=["team", "season"], how="left",
    )
    matchups = matchups.drop(columns=["team"], errors="ignore")

    # Differentials
    for col in ["tracking_avg_speed", "tracking_avg_dist", "tracking_off_def_speed_ratio"]:
        home_c = f"home_{col}"
        away_c = f"away_{col}"
        if home_c in matchups.columns and away_c in matchups.columns:
            matchups[f"diff_{col}"] = matchups[home_c] - matchups[away_c]

    return matchups
