"""
Roster quality features aggregated from individual player stats.

Uses players_advanced.csv from season N-1 combined with rosters.csv
from season N to measure each team's talent level.
"""

import numpy as np
import pandas as pd

from nba_predict.data.loader import load_players_advanced, load_rosters


def build_roster_quality() -> pd.DataFrame:
    """Build per-team roster quality metrics from prior-season player stats.

    Returns DataFrame with: team, season (prediction season),
    and aggregated roster quality columns.
    """
    players = load_players_advanced()
    rosters = load_rosters()

    # Use prior-season player stats: season N stats → features for season N+1
    players = players.copy()
    players["pred_season"] = players["season"] + 1

    # For each team-season, get the players on the roster
    # Then look up their PRIOR season stats
    roster_keys = rosters[["team", "season", "player"]].drop_duplicates()

    # Join: roster (team, season, player) with player stats (pred_season = season)
    merged = roster_keys.merge(
        players[["player", "team", "pred_season", "per", "bpm", "vorp", "ws", "ws_per_48", "mp", "g"]],
        left_on=["player", "season"],
        right_on=["player", "pred_season"],
        how="left",
        suffixes=("", "_stats"),
    )

    # Use the roster's team, not the player's prior-season team
    # (player may have been traded)
    merged["team_final"] = merged["team"]

    # Aggregate per team-season
    def _agg_team(group: pd.DataFrame) -> pd.Series:
        # Sort by minutes played to identify top players
        valid = group.dropna(subset=["mp"])
        valid = valid.sort_values("mp", ascending=False)

        top5 = valid.head(5)
        top8 = valid.head(8)

        return pd.Series({
            "roster_top5_bpm": top5["bpm"].mean() if len(top5) > 0 else np.nan,
            "roster_top5_per": top5["per"].mean() if len(top5) > 0 else np.nan,
            "roster_total_vorp": valid["vorp"].sum() if len(valid) > 0 else np.nan,
            "roster_total_ws": valid["ws"].sum() if len(valid) > 0 else np.nan,
            "roster_best_bpm": valid["bpm"].max() if len(valid) > 0 else np.nan,
            "roster_depth_bpm": top8["bpm"].mean() if len(top8) > 0 else np.nan,
            "roster_size": len(group),
        })

    result = merged.groupby(["team_final", "season"]).apply(_agg_team, include_groups=False).reset_index()
    result = result.rename(columns={"team_final": "team"})

    # Add roster continuity (% of players from prior season)
    continuity = _compute_roster_continuity(rosters)
    result = result.merge(continuity, on=["team", "season"], how="left")

    # Add average experience and age from roster
    roster_demo = _compute_roster_demographics(rosters)
    result = result.merge(roster_demo, on=["team", "season"], how="left")

    return result


def _compute_roster_continuity(rosters: pd.DataFrame) -> pd.DataFrame:
    """Compute the percentage of players retained from prior season."""
    records = []
    for (team, season), group in rosters.groupby(["team", "season"]):
        prior = rosters[(rosters["team"] == team) & (rosters["season"] == season - 1)]
        if len(prior) == 0:
            continuity = np.nan
        else:
            current_players = set(group["player"].dropna())
            prior_players = set(prior["player"].dropna())
            if len(prior_players) > 0:
                continuity = len(current_players & prior_players) / len(prior_players)
            else:
                continuity = np.nan
        records.append({"team": team, "season": season, "roster_continuity": continuity})
    return pd.DataFrame(records)


def _compute_roster_demographics(rosters: pd.DataFrame) -> pd.DataFrame:
    """Compute average experience and height per team-season."""
    agg = rosters.groupby(["team", "season"]).agg(
        roster_avg_exp=("exp", "mean"),
        roster_avg_height=("height_inches", "mean"),
    ).reset_index()
    return agg
