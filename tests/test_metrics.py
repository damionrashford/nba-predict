"""Tests for evaluation metrics."""

import numpy as np
import pytest

from nba_predict.evaluation.metrics import (
    calibration_metrics,
    classification_metrics,
    regression_metrics,
)


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        result = classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0

    def test_worst_predictions(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        result = classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0

    def test_with_probabilities(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        result = classification_metrics(y_true, y_pred, y_prob)
        assert "auc_roc" in result
        assert "log_loss" in result
        assert result["auc_roc"] == 1.0

    def test_n_samples(self):
        y_true = np.array([1, 0, 1])
        y_pred = np.array([1, 0, 0])
        result = classification_metrics(y_true, y_pred)
        assert result["n_samples"] == 3


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = regression_metrics(y_true, y_pred)
        assert result["mae"] == 0.0
        assert result["rmse"] == 0.0

    def test_known_error(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 21.0, 31.0])
        result = regression_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(1.0)
        assert result["rmse"] == pytest.approx(1.0)

    def test_n_samples(self):
        result = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert result["n_samples"] == 2


class TestCalibrationMetrics:
    def test_perfect_calibration(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        result = calibration_metrics(y_true, y_prob)
        assert result["brier_score"] == 0.0

    def test_returns_both_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.6, 0.4, 0.7, 0.3])
        result = calibration_metrics(y_true, y_prob)
        assert "brier_score" in result
        assert "ece" in result
        assert result["brier_score"] >= 0
        assert result["ece"] >= 0
