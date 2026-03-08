"""Tests for baseline models."""

import numpy as np

from nba_predict.evaluation.baselines import (
    always_home_baseline,
    constant_spread_baseline,
    last_season_baseline,
)


class TestAlwaysHomeBaseline:
    def test_mostly_home_wins(self):
        y_true = np.array([1, 1, 1, 0])
        result = always_home_baseline(y_true)
        assert result["accuracy"] == 0.75

    def test_mostly_away_wins(self):
        y_true = np.array([0, 0, 0, 1])
        result = always_home_baseline(y_true)
        assert result["accuracy"] == 0.25

    def test_has_name(self):
        result = always_home_baseline(np.array([1, 0]))
        assert result["name"] == "Always Home"


class TestConstantSpreadBaseline:
    def test_default_spread(self):
        y_true = np.array([5.0, -3.0, 10.0])
        result = constant_spread_baseline(y_true)
        assert result["name"] == "Constant +3.5"
        assert result["mae"] > 0

    def test_custom_spread(self):
        y_true = np.array([5.0, 5.0, 5.0])
        result = constant_spread_baseline(y_true, spread=5.0)
        assert result["mae"] == 0.0


class TestLastSeasonBaseline:
    def test_perfect_prior(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_prior = np.array([10.0, 20.0, 30.0])
        result = last_season_baseline(y_true, y_prior)
        assert result["mae"] == 0.0

    def test_handles_nan(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_prior = np.array([10.0, np.nan, 30.0])
        result = last_season_baseline(y_true, y_prior)
        assert result["mae"] == 0.0
        assert result["n_samples"] == 2
