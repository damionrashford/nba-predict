"""Data cleaning utilities for Basketball Reference data."""

import re

import numpy as np
import pandas as pd


def parse_bbref_date(date_str: str) -> pd.Timestamp | None:
    """Parse Basketball Reference date format: 'Tue, Nov 2, 1999' or similar."""
    if not isinstance(date_str, str) or not date_str.strip():
        return None
    try:
        return pd.to_datetime(date_str, format="mixed", dayfirst=False)
    except (ValueError, TypeError):
        return None


def parse_height_to_inches(height_str: str) -> float | None:
    """Convert '6-5' format to total inches (77)."""
    if not isinstance(height_str, str):
        return None
    match = re.match(r"(\d+)-(\d+)", height_str.strip())
    if match:
        return int(match.group(1)) * 12 + int(match.group(2))
    return None


def parse_game_streak(streak_str: str) -> int:
    """Convert 'W 5' → +5, 'L 3' → -3, '' → 0."""
    if not isinstance(streak_str, str) or not streak_str.strip():
        return 0
    streak_str = streak_str.strip()
    match = re.match(r"([WL])\s+(\d+)", streak_str)
    if match:
        sign = 1 if match.group(1) == "W" else -1
        return sign * int(match.group(2))
    return 0


def parse_experience(exp_str: str) -> int:
    """Convert years_experience: 'R' → 0, '5' → 5."""
    if not isinstance(exp_str, str):
        return 0
    exp_str = exp_str.strip()
    if exp_str.upper() == "R":
        return 0
    try:
        return int(exp_str)
    except ValueError:
        return 0


def parse_overtime(ot_str: str) -> int:
    """Convert overtime string: '' → 0, 'OT' → 1, '2OT' → 2, etc."""
    if not isinstance(ot_str, str) or not ot_str.strip():
        return 0
    ot_str = ot_str.strip().upper()
    if ot_str == "OT":
        return 1
    match = re.match(r"(\d+)OT", ot_str)
    if match:
        return int(match.group(1))
    return 0


def safe_float(series: pd.Series) -> pd.Series:
    """Convert a column to float, coercing errors to NaN. Strips '+' prefix."""
    return pd.to_numeric(series.astype(str).str.strip().str.lstrip("+"), errors="coerce")


def strip_multi_team_rows(df: pd.DataFrame, team_col: str = "team_name_abbr") -> pd.DataFrame:
    """Remove aggregate rows for players traded mid-season (2TM, 3TM, etc.)."""
    multi_team = {"2TM", "3TM", "4TM", "5TM", "TOT"}
    mask = ~df[team_col].astype(str).str.upper().isin(multi_team)
    return df[mask].copy()


def clean_team_column(series: pd.Series) -> pd.Series:
    """Strip asterisks and whitespace from a team name column."""
    return series.astype(str).str.strip().str.rstrip("*").str.strip()
