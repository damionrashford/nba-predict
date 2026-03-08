"""Tests for team name normalization."""

import pytest

from nba_predict.data.team_mapping import (
    CANONICAL_TEAMS,
    normalize_team_name,
)


class TestCanonicalTeams:
    def test_30_teams(self):
        assert len(CANONICAL_TEAMS) == 30

    def test_known_teams_present(self):
        for team in ["LAL", "BOS", "GSW", "CHI", "MIA", "BRK", "NYK"]:
            assert team in CANONICAL_TEAMS


class TestNormalizeTeamName:
    def test_canonical_passthrough(self):
        assert normalize_team_name("LAL") == "LAL"
        assert normalize_team_name("BOS") == "BOS"

    def test_full_name(self):
        assert normalize_team_name("Los Angeles Lakers") == "LAL"
        assert normalize_team_name("Boston Celtics") == "BOS"

    def test_historical_abbreviation(self):
        assert normalize_team_name("NJN") == "BRK"
        assert normalize_team_name("SEA") == "OKC"
        assert normalize_team_name("VAN") == "MEM"

    def test_playoff_asterisk(self):
        assert normalize_team_name("Los Angeles Lakers*") == "LAL"

    def test_whitespace(self):
        assert normalize_team_name("  LAL  ") == "LAL"

    def test_case_insensitive_abbr(self):
        assert normalize_team_name("lal") == "LAL"

    def test_unknown_team_raises(self):
        with pytest.raises(ValueError, match="Unknown team"):
            normalize_team_name("FAKE_TEAM")

    def test_all_historical_mappings(self):
        assert normalize_team_name("CHH") == "CHO"
        assert normalize_team_name("NOH") == "NOP"
        assert normalize_team_name("NOK") == "NOP"
        assert normalize_team_name("CHA") == "CHO"
