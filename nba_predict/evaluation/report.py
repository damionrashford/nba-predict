"""
Evaluation report generator.

Produces a markdown report summarizing all model performance,
baselines, and feature importances. Optionally generates charts.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nba_predict.config import REPORTS_DIR


def generate_report(all_results: dict[str, dict]) -> Path:
    """Generate a markdown evaluation report from training results.

    Args:
        all_results: Dict from pipeline.train_all() with per-model results.

    Returns:
        Path to the generated report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_report_{timestamp}.md"

    lines = []
    _add = lines.append

    _add("# NBA Prediction System — Evaluation Report")
    _add(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _add("---\n")

    # ─── Model 1: Game Winner ────────────────────────────────────────────
    if "game_winner" in all_results and "error" not in all_results["game_winner"]:
        r = all_results["game_winner"]
        _add("## Model 1: Game Winner (Classification)\n")
        _add("Predicts whether the home team wins.\n")

        metrics = r["test_metrics"]
        _add("### Test Set Performance\n")
        _add("| Metric | Value |")
        _add("|--------|-------|")
        _add(f"| Accuracy | {metrics['accuracy']:.4f} |")
        if "auc_roc" in metrics:
            _add(f"| AUC-ROC | {metrics['auc_roc']:.4f} |")
        if "log_loss" in metrics:
            _add(f"| Log Loss | {metrics['log_loss']:.4f} |")
        _add(f"| Samples | {metrics['n_samples']:,} |")

        _add("\n### Baselines\n")
        _add("| Baseline | Accuracy |")
        _add("|----------|----------|")
        for bl_name, bl_data in r.get("baselines", {}).items():
            _add(f"| {bl_data.get('name', bl_name)} | {bl_data['accuracy']:.4f} |")

        _add(_format_feature_importance(r.get("feature_importance")))
        _add("")

    # ─── Model 2: Point Spread ───────────────────────────────────────────
    if "point_spread" in all_results and "error" not in all_results["point_spread"]:
        r = all_results["point_spread"]
        _add("## Model 2: Point Spread (Regression)\n")
        _add("Predicts the margin of victory from the home team's perspective.\n")

        metrics = r["test_metrics"]
        _add("### Test Set Performance\n")
        _add("| Metric | Value |")
        _add("|--------|-------|")
        _add(f"| MAE | {metrics['mae']:.2f} pts |")
        _add(f"| RMSE | {metrics['rmse']:.2f} pts |")
        _add(f"| Samples | {metrics['n_samples']:,} |")

        if "derived_accuracy" in r:
            _add(f"\n**Derived winner accuracy** (sign of spread): {r['derived_accuracy']:.4f}")

        _add("\n### Baselines\n")
        _add("| Baseline | MAE |")
        _add("|----------|-----|")
        for bl_name, bl_data in r.get("baselines", {}).items():
            _add(f"| {bl_data.get('name', bl_name)} | {bl_data['mae']:.2f} |")

        _add(_format_feature_importance(r.get("feature_importance")))
        _add("")

    # ─── Model 3: Player Performance ─────────────────────────────────────
    if "player_performance" in all_results and "error" not in all_results["player_performance"]:
        r = all_results["player_performance"]
        _add("## Model 3: Player Performance (Regression)\n")
        _add("Predicts next-season per-game averages (PTS, AST, REB).\n")

        for target_name, target_results in r.get("results", {}).items():
            metrics = target_results["test_metrics"]
            bl = target_results.get("baseline", {})

            _add(f"### {target_name.upper()} per Game\n")
            _add("| Metric | Model | Baseline (Last Season) |")
            _add("|--------|-------|------------------------|")
            _add(f"| MAE | {metrics['mae']:.2f} | {bl.get('mae', 'N/A'):.2f} |")
            _add(f"| RMSE | {metrics['rmse']:.2f} | — |")

            improvement = target_results.get("improvement_pct", 0)
            _add(f"\n**Improvement over baseline:** {improvement:+.1f}%\n")

        _add("")

    # ─── Model 4: Season Outcomes ────────────────────────────────────────
    if "season_outcomes" in all_results and "error" not in all_results["season_outcomes"]:
        r = all_results["season_outcomes"]
        sub_results = r.get("results", {})

        _add("## Model 4: Season Outcomes\n")

        # 4A Win Totals
        if "win_totals" in sub_results:
            wt = sub_results["win_totals"]
            metrics = wt["test_metrics"]
            bl = wt.get("baseline", {})

            _add("### 4A: Win Totals\n")
            _add("Predicts total regular season wins per team.\n")
            _add("| Metric | Model | Baseline (Last Season) |")
            _add("|--------|-------|------------------------|")
            _add(f"| MAE | {metrics['mae']:.2f} wins | {bl.get('mae', 'N/A'):.2f} wins |")
            _add(f"| RMSE | {metrics['rmse']:.2f} wins | — |")

            improvement = wt.get("improvement_pct", 0)
            _add(f"\n**Improvement over baseline:** {improvement:+.1f}%\n")

        # 4B Championship Odds (just show example predictions)
        if "championship_odds" in sub_results:
            _add("### 4B: Championship Odds\n")
            _add("Derived from predicted win totals + prior SRS. See training output for per-season rankings.\n")

        # 4C MVP Race
        if "mvp_race" in sub_results:
            mvp = sub_results["mvp_race"]
            metrics = mvp["test_metrics"]

            _add("### 4C: MVP Race\n")
            _add("Predicts MVP award share (0-1) from individual + team stats.\n")
            _add("| Metric | Value |")
            _add("|--------|-------|")
            _add(f"| MAE | {metrics['mae']:.4f} |")
            _add(f"| RMSE | {metrics['rmse']:.4f} |")
            if not np.isnan(mvp.get("spearman_rho", np.nan)):
                _add(f"| Spearman Rank Corr | {mvp['spearman_rho']:.4f} |")

            _add(_format_feature_importance(mvp.get("feature_importance")))
        _add("")

    # ─── Summary ─────────────────────────────────────────────────────────
    _add("---\n")
    _add("## Methodology Notes\n")
    _add("- **Temporal split**: Train 2001-2021, Validation 2022-2023, Test 2024-2025")
    _add("- **Anti-leakage**: Rolling features use `.shift(1)`; "
         "team stats joined from prior season only")
    _add("- **Algorithm**: XGBoost (gradient-boosted decision trees) for all models")
    _add("- **Missing values**: XGBoost handles natively (learns optimal split direction)")
    _add("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print(f"\n  Report saved: {report_path}")
    return report_path


def _format_feature_importance(fi: pd.DataFrame | None) -> str:
    """Format feature importance as a markdown table."""
    if fi is None or fi.empty:
        return ""

    lines = ["\n### Top Features\n", "| Rank | Feature | Importance |", "|------|---------|------------|"]
    for i, (_, row) in enumerate(fi.head(10).iterrows(), 1):
        lines.append(f"| {i} | `{row['feature']}` | {row['importance']:.4f} |")
    return "\n".join(lines)
