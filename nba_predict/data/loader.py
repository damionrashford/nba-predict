"""Load and normalize all NBA CSV datasets."""

import pandas as pd

from nba_predict.config import DATA_DIR
from nba_predict.data.cleaning import (
    clean_team_column,
    parse_bbref_date,
    parse_experience,
    parse_game_streak,
    parse_height_to_inches,
    parse_overtime,
    safe_float,
    strip_multi_team_rows,
)
from nba_predict.data.team_mapping import normalize_team_name


def _normalize_col(series: pd.Series) -> pd.Series:
    """Apply normalize_team_name to a Series, dropping rows that fail."""
    def _safe_normalize(val):
        try:
            return normalize_team_name(str(val))
        except ValueError:
            return None
    return series.apply(_safe_normalize)


# ─── Individual Loaders ──────────────────────────────────────────────────────

def load_schedules() -> pd.DataFrame:
    """Load schedules.csv — the game-by-game backbone for prediction."""
    df = pd.read_csv(DATA_DIR / "schedules.csv", low_memory=False)

    # Team normalization
    df["team"] = _normalize_col(df["team"])
    df["opp_abbr"] = _normalize_col(df["opp_name"])

    # Parse date
    df["date"] = df["date_game"].apply(parse_bbref_date)

    # Binary home indicator
    df["is_home"] = df["game_location"].fillna("").str.strip() != "@"

    # Game result as integer (1=win, 0=loss)
    df["win"] = (df["game_result"].astype(str).str.strip().str.upper() == "W").astype(int)

    # Numeric columns
    for col in ["pts", "opp_pts", "wins", "losses"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    # Parse streak and overtime
    df["streak"] = df["game_streak"].apply(parse_game_streak)
    df["ot_count"] = df["overtimes"].apply(parse_overtime)

    # Season as int
    df["season"] = safe_float(df["season"]).astype("Int64")

    # Drop rows with no date or team (unplayable/future)
    df = df.dropna(subset=["date", "team", "opp_abbr", "season"])

    # Game number within team-season (1-indexed)
    df = df.sort_values(["team", "season", "date"])
    df["game_num"] = df.groupby(["team", "season"]).cumcount() + 1

    return df


def load_teams_advanced() -> pd.DataFrame:
    """Load teams_advanced.csv — efficiency metrics per team per season."""
    df = pd.read_csv(DATA_DIR / "teams_advanced.csv", low_memory=False)
    df["team_raw"] = clean_team_column(df["team"])
    df["team"] = _normalize_col(df["team_raw"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df = df.dropna(subset=["team", "season"])

    # Key numeric columns
    for col in ["off_rtg", "def_rtg", "net_rtg", "pace", "ts_pct",
                 "srs", "sos", "mov", "wins_pyth", "losses_pyth",
                 "orb_pct", "drb_pct", "trb_pct"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    return df


def load_teams_per_game() -> pd.DataFrame:
    """Load teams_per_game.csv — offensive per-game stats per team per season."""
    df = pd.read_csv(DATA_DIR / "teams_per_game.csv", low_memory=False)
    df["team_raw"] = clean_team_column(df["team"])
    df["team"] = _normalize_col(df["team_raw"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df = df.dropna(subset=["team", "season"])

    for col in ["fg_pct", "fg3_pct", "ft_pct", "trb", "ast", "stl",
                 "blk", "tov", "pts"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    return df


def load_teams_opponent() -> pd.DataFrame:
    """Load teams_opponent.csv — defensive per-game stats per team per season."""
    df = pd.read_csv(DATA_DIR / "teams_opponent.csv", low_memory=False)
    df["team_raw"] = clean_team_column(df["team"])
    df["team"] = _normalize_col(df["team_raw"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df = df.dropna(subset=["team", "season"])

    for col in df.columns:
        if col not in ("team", "team_raw", "season", "ranker"):
            df[col] = safe_float(df[col])

    return df


def load_players_advanced() -> pd.DataFrame:
    """Load players_advanced.csv — PER, BPM, VORP, WS per player per season."""
    df = pd.read_csv(DATA_DIR / "players_advanced.csv", low_memory=False)
    df = strip_multi_team_rows(df)
    df["team"] = _normalize_col(df["team_name_abbr"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df["player"] = df["name_display"].astype(str).str.strip()
    df = df.dropna(subset=["team", "season", "player"])

    # Standardize column name: CSV uses "games", we want "g"
    if "games" in df.columns:
        df = df.rename(columns={"games": "g"})

    for col in ["per", "ts_pct", "bpm", "obpm", "dbpm", "vorp",
                 "ws", "ws_per_48", "usg_pct", "mp", "g", "age"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    return df


def load_players_per_game() -> pd.DataFrame:
    """Load players_per_game.csv — per-game individual stats."""
    df = pd.read_csv(DATA_DIR / "players_per_game.csv", low_memory=False)
    df = strip_multi_team_rows(df)
    df["team"] = _normalize_col(df["team_name_abbr"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df["player"] = df["name_display"].astype(str).str.strip()
    df = df.dropna(subset=["team", "season", "player"])

    # Standardize column name: CSV uses "games", we want "g"
    if "games" in df.columns:
        df = df.rename(columns={"games": "g"})

    for col in ["pts_per_g", "ast_per_g", "trb_per_g", "stl_per_g",
                 "blk_per_g", "mp_per_g", "fg_pct", "fg3_pct", "ft_pct",
                 "tov_per_g", "g", "age"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    return df


def load_rosters() -> pd.DataFrame:
    """Load rosters.csv — team composition per season."""
    df = pd.read_csv(DATA_DIR / "rosters.csv", low_memory=False)
    df["team"] = _normalize_col(df["team"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df = df.dropna(subset=["team", "season"])

    df["height_inches"] = df["height"].apply(parse_height_to_inches)
    df["weight_num"] = safe_float(df["weight"])
    df["exp"] = df["years_experience"].apply(parse_experience)

    return df


def load_standings() -> pd.DataFrame:
    """Load standings.csv — final season records."""
    df = pd.read_csv(DATA_DIR / "standings.csv", low_memory=False)
    df["team"] = _normalize_col(df["team_name"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df = df.dropna(subset=["team", "season"])
    return df


def load_awards() -> pd.DataFrame:
    """Load awards.csv — MVP votes, All-Star, etc."""
    df = pd.read_csv(DATA_DIR / "awards.csv", low_memory=False)
    if "team_id" in df.columns:
        df["team"] = _normalize_col(df["team_id"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    df["player"] = df.get("player", pd.Series(dtype=str))

    for col in ["award_share", "votes_first", "points_won", "points_max", "age"]:
        if col in df.columns:
            df[col] = safe_float(df[col])

    return df


def load_draft() -> pd.DataFrame:
    """Load draft.csv — rookie draft picks."""
    df = pd.read_csv(DATA_DIR / "draft.csv", low_memory=False)
    if "team_id" in df.columns:
        df["team"] = _normalize_col(df["team_id"])
    df["season"] = safe_float(df["season"]).astype("Int64")
    return df


def load_sentiment() -> pd.DataFrame:
    """Load sentiment.csv — daily aggregated social sentiment per team."""
    from nba_predict.data.sentiment import load_sentiment as _load
    return _load()


# ─── Aggregate Loader ────────────────────────────────────────────────────────

def load_all() -> dict[str, pd.DataFrame]:
    """Load all datasets, returning a dict keyed by name."""
    return {
        "schedules": load_schedules(),
        "teams_advanced": load_teams_advanced(),
        "teams_per_game": load_teams_per_game(),
        "teams_opponent": load_teams_opponent(),
        "players_advanced": load_players_advanced(),
        "players_per_game": load_players_per_game(),
        "rosters": load_rosters(),
        "standings": load_standings(),
        "awards": load_awards(),
        "draft": load_draft(),
        "sentiment": load_sentiment(),
    }
