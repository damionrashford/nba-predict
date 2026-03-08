"""
Injury-based features for game-level predictions.

Aggregates injury report data into per-team-per-game features:
- Number of players listed as Out
- Number of players with injury (vs rest/G-League)
- Injury burden (weighted by player quality if available)

Injury data covers seasons 2022-2026. Historical seasons will have NaN
for these features, which XGBoost handles natively.
"""

import pandas as pd

from nba_predict.config import DATA_DIR
from nba_predict.data.team_mapping import normalize_team_name


def load_injuries() -> pd.DataFrame:
    """Load and normalize injuries.csv."""
    df = pd.read_csv(DATA_DIR / "injuries.csv", low_memory=False)

    # Extract team abbreviation from matchup field (e.g. "BKN@MIL")
    # The team_full column has full names; matchup has abbreviations
    # We need to map team_full -> canonical abbreviation
    def _extract_team_abbr(row):
        matchup = str(row.get("matchup", ""))
        team_full = str(row.get("team_full", ""))
        if "@" in matchup:
            away, home = matchup.split("@")
            # Determine which side this team is on
            # team_full is the team this player belongs to
            try:
                away_norm = normalize_team_name(away.strip())
                home_norm = normalize_team_name(home.strip())
            except ValueError:
                return None
            # Try to match team_full to one side
            try:
                team_norm = normalize_team_name(team_full.strip())
                return team_norm
            except ValueError:
                return away_norm  # fallback
        return None

    df["team"] = df.apply(_extract_team_abbr, axis=1)
    df["date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["team", "date"])

    return df


def build_injury_features() -> pd.DataFrame:
    """Build per-team-per-game injury features.

    Returns DataFrame with: team, date, injury_* columns.
    """
    inj = load_injuries()

    # Aggregate per team per game date
    agg = inj.groupby(["team", "date"]).agg(
        injury_players_out=("is_out", "sum"),
        injury_players_questionable=("is_questionable", "sum"),
        injury_players_doubtful=("is_doubtful", "sum"),
        injury_count_total=("is_out", "count"),
        injury_real_injuries=("is_injury", "sum"),
        injury_rest_dnp=("is_rest", "sum"),
        injury_gleague=("is_gleague", "sum"),
    ).reset_index()

    # Derived: injury severity score (weighted sum)
    agg["injury_severity"] = (
        agg["injury_players_out"] * 1.0 +
        agg["injury_players_doubtful"] * 0.7 +
        agg["injury_players_questionable"] * 0.3
    )

    return agg


def join_injury_features(matchups: pd.DataFrame,
                          injury_feats: pd.DataFrame) -> pd.DataFrame:
    """Join injury features onto matchup dataset for home and away teams."""
    inj = injury_feats.copy()
    matchups = matchups.copy()

    # Ensure date types match
    matchups["date"] = pd.to_datetime(matchups["date"], errors="coerce")
    inj["date"] = pd.to_datetime(inj["date"], errors="coerce")

    # Join for home team
    home_inj = inj.copy()
    home_rename = {c: f"home_{c}" for c in home_inj.columns if c not in ("team", "date")}
    home_inj = home_inj.rename(columns=home_rename)
    matchups = matchups.merge(
        home_inj, left_on=["team_home", "date"], right_on=["team", "date"], how="left",
    )
    matchups = matchups.drop(columns=["team"], errors="ignore")

    # Join for away team
    away_inj = inj.copy()
    away_rename = {c: f"away_{c}" for c in away_inj.columns if c not in ("team", "date")}
    away_inj = away_inj.rename(columns=away_rename)
    matchups = matchups.merge(
        away_inj, left_on=["team_away", "date"], right_on=["team", "date"], how="left",
    )
    matchups = matchups.drop(columns=["team"], errors="ignore")

    # Differential: how much worse off is the home team vs away?
    for col in ["injury_players_out", "injury_severity", "injury_real_injuries"]:
        home_c = f"home_{col}"
        away_c = f"away_{col}"
        if home_c in matchups.columns and away_c in matchups.columns:
            matchups[f"diff_{col}"] = matchups[home_c] - matchups[away_c]

    return matchups
