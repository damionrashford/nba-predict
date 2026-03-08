"""
Sentiment data loading and processing for the prediction pipeline.

Loads pre-aggregated daily sentiment scores (from collect_sentiment.py)
and merges them onto the game schedule with a 1-day lookback to prevent
leakage from game-day posts that may contain results.
"""

import pandas as pd

from nba_predict.config import DATA_DIR

# Expected columns in the aggregated CSV
_SENTIMENT_COLS = [
    "sentiment_score",
    "sentiment_volume",
    "sentiment_std",
    "sentiment_engagement",
    "sentiment_pos_ratio",
]


def load_sentiment() -> pd.DataFrame:
    """Load aggregated daily sentiment data.

    Returns DataFrame with columns:
        team: str (canonical 3-letter code)
        date: pd.Timestamp
        sentiment_score: float [-1, 1]
        sentiment_volume: int
        sentiment_std: float
        sentiment_engagement: float
        sentiment_pos_ratio: float [0, 1]

    Returns empty DataFrame with correct schema if no data file exists.
    """
    path = DATA_DIR / "sentiment.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "date"] + _SENTIMENT_COLS)

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["team", "date"] + _SENTIMENT_COLS)

    df["date"] = pd.to_datetime(df["date"])

    # Ensure numeric types
    for col in _SENTIMENT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["team", "date"] + [c for c in _SENTIMENT_COLS if c in df.columns]]


def merge_sentiment_to_games(games: pd.DataFrame,
                             sentiment: pd.DataFrame) -> pd.DataFrame:
    """Left-join daily sentiment onto the games DataFrame.

    Uses a 1-day lookback: for each game on date D, uses sentiment
    from date D-1 (the day BEFORE the game). This prevents leakage
    from game-day posts that contain spoilers/results.

    The subsequent .shift(1) in game_features.py provides a second
    layer of anti-leakage protection.

    Args:
        games: schedules DataFrame with 'team' and 'date' columns.
        sentiment: output of load_sentiment().

    Returns:
        games DataFrame with sentiment columns appended (NaN where no data).
    """
    if sentiment.empty:
        # Add empty sentiment columns so downstream code doesn't break
        for col in _SENTIMENT_COLS:
            games[col] = float("nan")
        return games

    # Shift sentiment date forward by 1 day so it aligns with the NEXT game day.
    # Game on March 5 → uses sentiment from March 4.
    sent = sentiment.copy()
    sent["date"] = sent["date"] + pd.Timedelta(days=1)

    result = games.merge(sent, on=["team", "date"], how="left", suffixes=("", "_sent"))

    # Drop any duplicate columns from merge
    result = result.drop(
        columns=[c for c in result.columns if c.endswith("_sent")],
        errors="ignore",
    )

    return result
