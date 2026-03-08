"""Tests for data cleaning utilities."""

import numpy as np
import pandas as pd
import pytest

from nba_predict.data.cleaning import (
    parse_bbref_date,
    parse_experience,
    parse_game_streak,
    parse_height_to_inches,
    parse_overtime,
    safe_float,
    strip_multi_team_rows,
)


class TestParseBbrefDate:
    def test_standard_format(self):
        result = parse_bbref_date("Tue, Nov 2, 1999")
        assert result is not None
        assert result.year == 1999
        assert result.month == 11
        assert result.day == 2

    def test_empty_string(self):
        assert parse_bbref_date("") is None

    def test_none_input(self):
        assert parse_bbref_date(None) is None

    def test_non_string(self):
        assert parse_bbref_date(12345) is None


class TestParseHeightToInches:
    def test_standard(self):
        assert parse_height_to_inches("6-5") == 77

    def test_seven_footer(self):
        assert parse_height_to_inches("7-0") == 84

    def test_short_player(self):
        assert parse_height_to_inches("5-9") == 69

    def test_none_input(self):
        assert parse_height_to_inches(None) is None

    def test_invalid_format(self):
        assert parse_height_to_inches("tall") is None


class TestParseGameStreak:
    def test_win_streak(self):
        assert parse_game_streak("W 5") == 5

    def test_loss_streak(self):
        assert parse_game_streak("L 3") == -3

    def test_empty(self):
        assert parse_game_streak("") == 0

    def test_none(self):
        assert parse_game_streak(None) == 0


class TestParseExperience:
    def test_rookie(self):
        assert parse_experience("R") == 0

    def test_veteran(self):
        assert parse_experience("5") == 5

    def test_none(self):
        assert parse_experience(None) == 0


class TestParseOvertime:
    def test_no_ot(self):
        assert parse_overtime("") == 0
        assert parse_overtime(None) == 0

    def test_single_ot(self):
        assert parse_overtime("OT") == 1

    def test_double_ot(self):
        assert parse_overtime("2OT") == 2

    def test_triple_ot(self):
        assert parse_overtime("3OT") == 3


class TestSafeFloat:
    def test_normal_numbers(self):
        s = pd.Series(["1.5", "2.3", "3.0"])
        result = safe_float(s)
        assert result.tolist() == [1.5, 2.3, 3.0]

    def test_plus_prefix(self):
        s = pd.Series(["+5.2", "+3.1"])
        result = safe_float(s)
        assert result.tolist() == [5.2, 3.1]

    def test_invalid_values(self):
        s = pd.Series(["abc", "1.0", ""])
        result = safe_float(s)
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 1.0


class TestStripMultiTeamRows:
    def test_removes_multi_team(self):
        df = pd.DataFrame({"team_name_abbr": ["LAL", "2TM", "BOS", "TOT", "GSW"]})
        result = strip_multi_team_rows(df)
        assert list(result["team_name_abbr"]) == ["LAL", "BOS", "GSW"]

    def test_no_multi_team(self):
        df = pd.DataFrame({"team_name_abbr": ["LAL", "BOS"]})
        result = strip_multi_team_rows(df)
        assert len(result) == 2
