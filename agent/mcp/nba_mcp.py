"""NBA Data Query MCP Server.

Exposes NBA dataset query tools for the fast-agent analyst.
Runs as a stdio MCP server. RAGs over 17 CSV datasets (2000–2026).

Tools:
  - query_team_stats        → team advanced + per-game stats
  - query_player_stats      → player per-game + advanced stats
  - query_head_to_head      → matchup history between two teams
  - query_standings         → full season standings
  - query_schedule          → game-by-game results for a team
  - list_teams              → all 30 team codes
  - list_seasons            → available seasons
"""

import sys
from pathlib import Path

# Add project root to path so we can import nba_predict
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from mcp.server.fastmcp import FastMCP

from nba_predict.config import LIVE_SEASON
from nba_predict.data.loader import (
    load_players_advanced,
    load_players_per_game,
    load_schedules,
    load_teams_advanced,
    load_teams_per_game,
)
from nba_predict.data.team_mapping import normalize_team_name

# ─── Server Setup ─────────────────────────────────────────────────────────────

mcp = FastMCP("NBA Data")


def _resolve_team(team_input: str) -> str:
    """Normalize team input (e.g., 'Lakers' → 'LAL', 'Boston' → 'BOS')."""
    return normalize_team_name(team_input)


# ─── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
def query_team_stats(team: str, season: int | None = None) -> str:
    """Look up team stats (advanced + per-game) for a specific team and season.

    Args:
        team: Team name or abbreviation
        season: Season year (optional, defaults to most recent)

    Returns:
        Team advanced stats and per-game averages.
    """
    team_code = _resolve_team(team)
    adv = load_teams_advanced()
    pg = load_teams_per_game()

    if season:
        adv = adv[adv["season"] == season]
        pg = pg[pg["season"] == season]
    else:
        season = int(adv["season"].max())
        adv = adv[adv["season"] == season]
        pg = pg[pg["season"] == season]

    team_adv = adv[adv["team"] == team_code]
    team_pg = pg[pg["team"] == team_code]

    if team_adv.empty:
        return f"No data for {team_code} in {season}."

    row = team_adv.iloc[0]
    lines = [
        f"**{team_code} — {season} Season Stats**\n",
        "### Advanced",
        f"- Record: {row.get('wins', '?')}-{row.get('losses', '?')}",
        f"- Off Rating: {row.get('off_rtg', '?')}",
        f"- Def Rating: {row.get('def_rtg', '?')}",
        f"- Net Rating: {row.get('net_rtg', '?')}",
        f"- SRS: {row.get('srs', '?')}",
        f"- Pace: {row.get('pace', '?')}",
        f"- TS%: {row.get('ts_pct', '?')}",
    ]

    if not team_pg.empty:
        pg_row = team_pg.iloc[0]
        lines += [
            "\n### Per Game",
            f"- PTS: {pg_row.get('pts', '?')}",
            f"- FG%: {pg_row.get('fg_pct', '?')}",
            f"- 3P%: {pg_row.get('fg3_pct', '?')}",
            f"- FT%: {pg_row.get('ft_pct', '?')}",
            f"- REB: {pg_row.get('trb', '?')}",
            f"- AST: {pg_row.get('ast', '?')}",
        ]

    return "\n".join(lines)


@mcp.tool()
def query_player_stats(player_name: str, season: int | None = None) -> str:
    """Look up player stats (per-game + advanced) for a specific player.

    Args:
        player_name: Player name (partial match supported)
        season: Season year (optional, defaults to most recent for that player)

    Returns:
        Player per-game and advanced stats.
    """
    pg = load_players_per_game()
    adv = load_players_advanced()

    matches = pg[pg["player"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        return f"Player not found: '{player_name}'"

    if season:
        matches = matches[matches["season"] == season]
    else:
        season = int(matches["season"].max())
        matches = matches[matches["season"] == season]

    if matches.empty:
        return f"No data for '{player_name}' in {season}."

    row = matches.iloc[0]
    player = row["player"]

    adv_match = adv[
        (adv["player"] == player) & (adv["season"] == season)
    ]

    lines = [
        f"**{player}** — {season} Season\n",
        f"Team: {row.get('team', '?')} | Age: {row.get('age', '?')} | "
        f"Games: {row.get('g', '?')}\n",
        "### Per Game",
        f"- PTS: {row.get('pts_per_g', '?')}",
        f"- AST: {row.get('ast_per_g', '?')}",
        f"- REB: {row.get('trb_per_g', '?')}",
        f"- STL: {row.get('stl_per_g', '?')}",
        f"- BLK: {row.get('blk_per_g', '?')}",
        f"- FG%: {row.get('fg_pct', '?')}",
        f"- 3P%: {row.get('fg3_pct', '?')}",
        f"- MP: {row.get('mp_per_g', '?')}",
    ]

    if not adv_match.empty:
        a = adv_match.iloc[0]
        lines += [
            "\n### Advanced",
            f"- PER: {a.get('per', '?')}",
            f"- BPM: {a.get('bpm', '?')}",
            f"- VORP: {a.get('vorp', '?')}",
            f"- WS: {a.get('ws', '?')}",
            f"- USG%: {a.get('usg_pct', '?')}",
            f"- TS%: {a.get('ts_pct', '?')}",
        ]

    return "\n".join(lines)


@mcp.tool()
def query_head_to_head(team1: str, team2: str, season: int | None = None) -> str:
    """Look up head-to-head results between two teams.

    Args:
        team1: First team name or abbreviation
        team2: Second team name or abbreviation
        season: Season year (optional, shows all if omitted)

    Returns:
        Game-by-game results between the two teams.
    """
    t1 = _resolve_team(team1)
    t2 = _resolve_team(team2)
    sched = load_schedules()

    games = sched[
        ((sched["team"] == t1) & (sched["opp_abbr"] == t2)) |
        ((sched["team"] == t2) & (sched["opp_abbr"] == t1))
    ]

    if season:
        games = games[games["season"] == season]

    # Deduplicate to one row per game (take home team perspective)
    games = games[games["is_home"]].sort_values("date", ascending=False)

    if games.empty:
        return f"No head-to-head games found for {t1} vs {t2}."

    lines = [f"**{t1} vs {t2}** — Head-to-Head\n"]

    # Show last 10 games
    for _, g in games.head(10).iterrows():
        home = g["team"]
        away = g["opp_abbr"]
        result = "W" if g["win"] else "L"
        lines.append(
            f"- {str(g['date'])[:10]}: {away} @ {home} — "
            f"{home} {result} ({g['pts']:.0f}-{g['opp_pts']:.0f})"
        )

    t1_wins = games[(games["team"] == t1) & games["win"]].shape[0]
    t2_wins = games[(games["team"] == t2) & games["win"]].shape[0]
    total = games.shape[0]
    lines.insert(
        1,
        f"Record (home perspective, last {total}): "
        f"{t1} {t1_wins}W / {t2} {t2_wins}W\n",
    )

    return "\n".join(lines)


@mcp.tool()
def query_standings(season: int | None = None) -> str:
    """Get the full NBA standings for a season.

    Args:
        season: Season year (defaults to most recent)

    Returns:
        Full standings sorted by wins.
    """
    adv = load_teams_advanced()

    if not season:
        season = int(adv["season"].max())

    season_df = adv[adv["season"] == season].copy()
    if season_df.empty:
        return f"No standings data for {season}."

    season_df["wins"] = pd.to_numeric(season_df["wins"], errors="coerce")
    season_df = season_df.sort_values("wins", ascending=False)

    lines = [
        f"**{season} NBA Standings**\n",
        "| Rank | Team | W | L | Win% | SRS | Off Rtg | Def Rtg |",
        "|------|------|---|---|------|-----|---------|---------|",
    ]
    for i, (_, row) in enumerate(season_df.iterrows(), 1):
        w = row.get("wins", 0)
        l = row.get("losses", 0)
        wp = w / max(w + l, 1)
        lines.append(
            f"| {i} | {row['team']} | {w:.0f} | {l:.0f} | {wp:.3f} | "
            f"{row.get('srs', '?')} | {row.get('off_rtg', '?')} | "
            f"{row.get('def_rtg', '?')} |"
        )

    return "\n".join(lines)


@mcp.tool()
def query_schedule(team: str, season: int | None = None) -> str:
    """Get game-by-game results for a team.

    Args:
        team: Team name or abbreviation
        season: Season year (defaults to most recent)

    Returns:
        Game log with dates, opponents, and scores.
    """
    team_code = _resolve_team(team)
    sched = load_schedules()

    team_games = sched[sched["team"] == team_code]
    if not season:
        season = int(team_games["season"].max())

    team_games = team_games[team_games["season"] == season].sort_values("date")

    if team_games.empty:
        return f"No schedule data for {team_code} in {season}."

    wins = team_games["win"].sum()
    losses = len(team_games) - wins

    lines = [
        f"**{team_code} — {season} Schedule** ({wins:.0f}-{losses:.0f})\n",
        "| Date | Loc | Opp | Result | Score |",
        "|------|-----|-----|--------|-------|",
    ]

    for _, g in team_games.iterrows():
        loc = "vs" if g["is_home"] else "@"
        result = "W" if g["win"] else "L"
        lines.append(
            f"| {str(g['date'])[:10]} | {loc} | {g['opp_abbr']} | "
            f"{result} | {g['pts']:.0f}-{g['opp_pts']:.0f} |"
        )

    return "\n".join(lines)


@mcp.tool()
def list_teams() -> str:
    """List all 30 NBA team codes used in the system.

    Returns:
        All canonical team abbreviations.
    """
    adv = load_teams_advanced()
    latest = int(adv["season"].max())
    teams = sorted(adv[adv["season"] == latest]["team"].unique())
    return (
        f"**NBA Teams ({latest} season):**\n\n"
        + ", ".join(teams)
        + f"\n\n{len(teams)} teams total"
    )


@mcp.tool()
def list_seasons() -> str:
    """List all available seasons in the dataset.

    Returns:
        Range of seasons with data availability notes.
    """
    adv = load_teams_advanced()
    seasons = sorted(adv["season"].unique())
    return (
        f"**Available Seasons:** {int(min(seasons))}–{int(max(seasons))}\n\n"
        f"- Training data: 2001–2021\n"
        f"- Validation: 2022–2023\n"
        f"- Test: 2024–2025\n"
        f"- Live (current): {LIVE_SEASON}\n"
    )


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
