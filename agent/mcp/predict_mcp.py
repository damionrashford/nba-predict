"""Prediction MCP Server.

Exposes trained XGBoost prediction models as MCP tools.
Runs as a stdio MCP server.

Tools:
  - predict_game_winner     → home win probability + prediction
  - predict_point_spread    → predicted margin
  - predict_player_stats    → next-season pts/ast/reb
  - predict_season_wins     → predicted win total
  - predict_mvp_race        → top MVP candidates with predicted award shares
"""

import sys
from pathlib import Path

# Add project root to path so we can import nba_predict
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from mcp.server.fastmcp import FastMCP

from nba_predict.config import LIVE_SEASON, MODELS_DIR
from nba_predict.data.team_mapping import normalize_team_name

# ─── Server Setup ─────────────────────────────────────────────────────────────

mcp = FastMCP("NBA Predict")

# ─── Lazy Model Loading ───────────────────────────────────────────────────────

_cache = {}


def _load_model(name: str) -> dict:
    """Load a saved model artifact, caching for reuse."""
    if name not in _cache:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Model '{name}' not found at {path}. "
                "Run 'python scripts/train.py' first."
            )
        _cache[name] = joblib.load(path)
    return _cache[name]


def _resolve_team(team_input: str) -> str:
    """Normalize team input (e.g., 'Lakers' → 'LAL', 'Boston' → 'BOS')."""
    return normalize_team_name(team_input)


# ─── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def predict_game_winner(home_team: str, away_team: str) -> str:
    """Predict the winner of an NBA game.

    Args:
        home_team: Home team name or abbreviation (e.g., 'LAL', 'Lakers', 'Los Angeles Lakers')
        away_team: Away team name or abbreviation

    Returns:
        Prediction with home win probability and key factors.
    """
    home = _resolve_team(home_team)
    away = _resolve_team(away_team)

    artifact = _load_model("game_winner")
    model = artifact["model"]
    calibrator = artifact.get("calibrator")
    feature_cols = artifact["feature_cols"]

    from nba_predict.features.matchup_features import build_matchup_dataset
    df = build_matchup_dataset()
    live = df[df["season"] == LIVE_SEASON]

    matchup = live[
        (live["team_home"] == home) & (live["team_away"] == away)
    ]
    if matchup.empty:
        matchup = live[
            (live["team_home"] == away) & (live["team_away"] == home)
        ]
        if matchup.empty:
            return (
                f"No matchup data found for {home} vs {away} "
                f"in {LIVE_SEASON} season."
            )

    latest = matchup.iloc[-1:]
    X = latest[feature_cols].astype(float)

    raw_prob = model.predict_proba(X)[:, 1][0]
    prob = calibrator.predict([raw_prob])[0] if calibrator else raw_prob
    winner = home if prob > 0.5 else away
    confidence = prob if prob > 0.5 else 1 - prob

    return (
        f"**{away} @ {home}**\n\n"
        f"Predicted Winner: **{winner}**\n"
        f"Home Win Probability: {prob:.1%}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Key context:\n"
        f"- Home record this season: "
        f"{latest['home_win_pct_season'].values[0]:.3f}\n"
        f"- Away record this season: "
        f"{latest['away_win_pct_season'].values[0]:.3f}\n"
    )


@mcp.tool()
def predict_point_spread(home_team: str, away_team: str) -> str:
    """Predict the point spread (margin) for an NBA game.

    Args:
        home_team: Home team name or abbreviation
        away_team: Away team name or abbreviation

    Returns:
        Predicted margin from home team's perspective.
    """
    home = _resolve_team(home_team)
    away = _resolve_team(away_team)

    artifact = _load_model("point_spread")
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    from nba_predict.features.matchup_features import build_matchup_dataset
    df = build_matchup_dataset()
    live = df[df["season"] == LIVE_SEASON]

    matchup = live[
        (live["team_home"] == home) & (live["team_away"] == away)
    ]
    if matchup.empty:
        matchup = live[
            (live["team_home"] == away) & (live["team_away"] == home)
        ]
        if matchup.empty:
            return (
                f"No matchup data found for {home} vs {away} "
                f"in {LIVE_SEASON} season."
            )

    latest = matchup.iloc[-1:]
    X = latest[feature_cols].astype(float)
    spread = model.predict(X)[0]

    sign = "+" if spread > 0 else ""
    favored = home if spread > 0 else away
    margin = abs(spread)

    return (
        f"**{away} @ {home}**\n\n"
        f"Predicted Spread: Home {sign}{spread:.1f}\n"
        f"Favored: **{favored}** by {margin:.1f} points\n"
    )


@mcp.tool()
def predict_player_stats(player_name: str) -> str:
    """Predict next-season per-game averages for an NBA player.

    Args:
        player_name: Player name (e.g., 'LeBron James', 'Curry')

    Returns:
        Predicted PTS, AST, REB per game with comparison to last season.
    """
    artifact = _load_model("player_performance")
    models = artifact["models"]
    feature_cols = artifact["feature_cols"]

    from nba_predict.models.player_performance import _build_player_dataset
    df = _build_player_dataset()

    player_df = df[
        df["player"].str.contains(player_name, case=False, na=False)
    ]
    if player_df.empty:
        return f"Player not found: '{player_name}'. Try a more specific name."

    latest = player_df.loc[player_df["season"].idxmax()]
    X = pd.DataFrame([latest[feature_cols].astype(float)])

    lines = [
        f"**{latest['player']}** — Predicted Next Season Stats",
        f"Based on {int(latest['feature_season'])} season\n",
        "| Stat | Predicted | Last Season |",
        "|------|-----------|-------------|",
    ]

    stat_map = {"pts": "pts_per_g", "ast": "ast_per_g", "reb": "trb_per_g"}
    for target_name, model in models.items():
        pred = model.predict(X)[0]
        prior = latest[stat_map[target_name]]
        lines.append(f"| {target_name.upper()} | {pred:.1f} | {prior:.1f} |")

    return "\n".join(lines)


@mcp.tool()
def predict_season_wins(team: str) -> str:
    """Predict total regular season wins for an NBA team.

    Args:
        team: Team name or abbreviation

    Returns:
        Predicted wins with prior season comparison.
    """
    team_code = _resolve_team(team)

    artifact = _load_model("season_outcomes")
    xgb_model = artifact["win_totals_model"]
    ridge = artifact.get("win_totals_ridge")
    scaler = artifact.get("win_totals_scaler")
    features = artifact["win_totals_features"]

    from nba_predict.models.season_outcomes import _build_win_totals_dataset
    df = _build_win_totals_dataset()

    team_df = df[df["team"] == team_code].sort_values("season")
    if team_df.empty:
        return f"Team not found: '{team}'"

    latest = team_df.iloc[-1:]
    X = latest[[c for c in features if c in latest.columns]].astype(float)

    xgb_pred = xgb_model.predict(X)[0]
    if ridge and scaler:
        X_scaled = scaler.transform(X.fillna(0))
        ridge_pred = ridge.predict(X_scaled)[0]
        pred_wins = 0.4 * xgb_pred + 0.6 * ridge_pred
    else:
        pred_wins = xgb_pred

    prev_wins = latest["prev_wins"].values[0]

    return (
        f"**{team_code}** — Season Win Prediction\n\n"
        f"Predicted Wins: **{pred_wins:.1f}**\n"
        f"Prior Season Wins: {prev_wins:.0f}\n"
        f"Projected Change: {pred_wins - float(prev_wins):+.1f} wins\n"
    )


@mcp.tool()
def predict_mvp_race(season: int | None = None) -> str:
    """Predict the MVP race — top candidates with predicted award shares.

    Args:
        season: Season year (optional, defaults to most recent).

    Returns:
        Top 10 MVP candidates ranked by predicted award share.
    """
    artifact = _load_model("season_outcomes")
    model = artifact["mvp_model"]
    features = artifact["mvp_features"]

    from nba_predict.models.season_outcomes import _build_mvp_dataset
    df = _build_mvp_dataset()

    if season:
        df = df[df["season"] == season]
    else:
        season = int(df["season"].max())
        df = df[df["season"] == season]

    if df.empty:
        return f"No MVP data for season {season}."

    X = df[[c for c in features if c in df.columns]].astype(float)
    df = df.copy()
    df["pred_award_share"] = model.predict(X)

    top10 = df.nlargest(10, "pred_award_share")

    lines = [
        f"**{season} MVP Race — Predicted Rankings**\n",
        "| Rank | Player | Pred Share | Team | PPG | BPM |",
        "|------|--------|-----------|------|-----|-----|",
    ]
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        lines.append(
            f"| {i} | {row['player']} | "
            f"{row['pred_award_share']:.3f} | "
            f"{row['team']} | {row.get('pts_per_g', 0):.1f} | "
            f"{row.get('bpm', 0):.1f} |"
        )

    return "\n".join(lines)


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
