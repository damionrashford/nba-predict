"""
Matchup-level feature construction.

Pairs home/away game rows into unique matchups, joins prior-season team
stats and roster quality, then computes differential features.
This produces the final training DataFrame for game-level models.
"""

import pandas as pd

from nba_predict.features.game_features import build_game_features, get_game_feature_columns
from nba_predict.features.team_features import build_prior_season_features, join_prior_season
from nba_predict.features.player_features import build_roster_quality
from nba_predict.data.loader import load_schedules
from nba_predict.config import FIRST_USABLE_SEASON


def build_matchup_dataset() -> pd.DataFrame:
    """Build the complete matchup dataset for game-level predictions.

    Each row is a unique game with home team perspective.
    Features come from rolling game stats, prior-season team stats,
    and roster quality — all computed without data leakage.

    Returns DataFrame with:
      - Identifiers: date, season, team_home, team_away
      - Features: ~50 columns (home_*, away_*, diff_*)
      - Targets: home_win (0/1), margin (pts difference)
    """
    # Step 1: Load schedules and build per-team rolling features
    schedules = load_schedules()
    schedules = build_game_features(schedules)

    # Step 2: Split into home and away games
    game_feat_cols = get_game_feature_columns()
    keep_cols = ["date", "season", "team", "opp_abbr", "win", "pts", "opp_pts", "margin"] + game_feat_cols

    home = schedules[schedules["is_home"]].copy()
    away = schedules[~schedules["is_home"]].copy()

    # Rename columns with home_/away_ prefix for the merge
    home_renamed = home[keep_cols].rename(
        columns={c: f"home_{c}" for c in keep_cols if c not in ("date", "season")}
    )
    away_renamed = away[keep_cols].rename(
        columns={c: f"away_{c}" for c in keep_cols if c not in ("date", "season")}
    )

    # Step 3: Pair home + away rows for the same game
    matchups = home_renamed.merge(
        away_renamed,
        left_on=["date", "home_team", "home_opp_abbr"],
        right_on=["date", "away_opp_abbr", "away_team"],
        suffixes=("", "_dup"),
    )

    # Use the home game's season
    matchups["season"] = matchups["season"].combine_first(matchups.get("season_dup"))
    matchups.drop(columns=[c for c in matchups.columns if c.endswith("_dup")], inplace=True)

    # Rename for clarity
    matchups = matchups.rename(columns={
        "home_team": "team_home",
        "away_team": "team_away",
    })

    # Targets
    matchups["home_win"] = matchups["home_win"].astype(int)
    matchups["margin"] = matchups["home_margin"]

    # Step 4: Join prior-season team stats
    prior_stats = build_prior_season_features()
    matchups = join_prior_season(matchups, prior_stats, "team_home", "home_")
    matchups = join_prior_season(matchups, prior_stats, "team_away", "away_")

    # Step 5: Join roster quality
    roster_qual = build_roster_quality()
    matchups = _join_roster_quality(matchups, roster_qual, "team_home", "home_")
    matchups = _join_roster_quality(matchups, roster_qual, "team_away", "away_")

    # Step 6: Build differential features
    matchups = _build_differentials(matchups)

    # Step 7: Filter to usable seasons (need prior-season data)
    matchups = matchups[matchups["season"] >= FIRST_USABLE_SEASON]

    return matchups


def _join_roster_quality(df: pd.DataFrame, roster_qual: pd.DataFrame,
                         team_col: str, prefix: str) -> pd.DataFrame:
    """Join roster quality features onto matchups."""
    rq = roster_qual.copy()
    rename_map = {col: f"{prefix}{col}" for col in rq.columns if col not in ("team", "season")}
    rq = rq.rename(columns=rename_map)

    result = df.merge(
        rq, left_on=[team_col, "season"], right_on=["team", "season"], how="left",
    )

    # Clean up duplicate team column
    if "team" in result.columns and team_col in result.columns and team_col != "team":
        result = result.drop(columns=["team"], errors="ignore")

    return result


def _build_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute home minus away differential for key features."""
    diff_pairs = [
        # Rolling game stats
        ("win_pct_season", "win_pct_season"),
        ("win_pct_last5", "win_pct_last5"),
        ("win_pct_last10", "win_pct_last10"),
        ("win_pct_last20", "win_pct_last20"),
        ("margin_avg_last5", "margin_avg_last5"),
        ("margin_avg_last10", "margin_avg_last10"),
        ("margin_avg_last20", "margin_avg_last20"),
        ("pts_avg_last10", "pts_avg_last10"),
        ("pts_avg_season", "pts_avg_season"),
        ("pts_allowed_avg_season", "pts_allowed_avg_season"),
        # Prior-season team quality
        ("srs_prev", "srs_prev"),
        ("off_rtg_prev", "off_rtg_prev"),
        ("def_rtg_prev", "def_rtg_prev"),
        ("net_rtg_prev", "net_rtg_prev"),
        ("pace_prev", "pace_prev"),
        # Four Factors differentials
        ("efg_pct_prev", "efg_pct_prev"),
        ("tov_pct_prev", "tov_pct_prev"),
        ("orb_pct_prev", "orb_pct_prev"),
        ("opp_efg_pct_prev", "opp_efg_pct_prev"),
        # Roster quality
        ("roster_top5_bpm", "roster_top5_bpm"),
        ("roster_total_vorp", "roster_total_vorp"),
        ("roster_best_bpm", "roster_best_bpm"),
        ("roster_depth_bpm", "roster_depth_bpm"),
        ("roster_total_ws", "roster_total_ws"),
        # Sentiment (NaN for historical seasons — XGBoost handles natively)
        ("sentiment_score_last10", "sentiment_score_last10"),
        ("sentiment_volume_last10", "sentiment_volume_last10"),
        ("sentiment_std_last10", "sentiment_std_last10"),
        ("sentiment_engagement_last10", "sentiment_engagement_last10"),
        ("sentiment_pos_ratio_last10", "sentiment_pos_ratio_last10"),
        ("sentiment_score_prev", "sentiment_score_prev"),
        ("sentiment_volume_prev", "sentiment_volume_prev"),
    ]

    for home_suffix, away_suffix in diff_pairs:
        home_col = f"home_{home_suffix}"
        away_col = f"away_{away_suffix}"
        diff_name = f"diff_{home_suffix}"
        if home_col in df.columns and away_col in df.columns:
            df[diff_name] = df[home_col] - df[away_col]

    # Rest advantage
    if "home_rest_days" in df.columns and "away_rest_days" in df.columns:
        df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return all feature column names (excluding identifiers and targets)."""
    exclude = {
        "date", "season", "team_home", "team_away",
        "home_win", "margin",
        "home_pts", "home_opp_pts", "home_margin",
        "away_pts", "away_opp_pts", "away_margin",
        "home_win_x", "away_win",
        "home_team", "away_team",
        "home_opp_abbr", "away_opp_abbr",
    }
    return [c for c in df.columns if c not in exclude and not c.endswith("_abbr")]
