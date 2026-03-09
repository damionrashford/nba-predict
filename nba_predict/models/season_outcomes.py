"""
Model 4: Season Outcomes (Multi-target).

Three sub-models predicting season-level outcomes:

  4A — Win Totals: Predict final regular season wins per team.
        XGBRegressor. Baseline: prior-season wins.

  4B — Championship Odds: Rank teams by predicted wins + SRS.
        Calibrated via historical win-to-championship mapping.
        No separate model — derived from 4A predictions + team quality.

  4C — MVP Race: Predict award_share (0-1) for MVP voting.
        XGBRegressor on player-season features.
        Eval: MAE + Spearman rank correlation with actual results.
"""

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from nba_predict.config import (
    MODELS_DIR, RANDOM_SEED, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS,
    XGBOOST_REGRESSOR_PARAMS,
)
from nba_predict.data.loader import (
    load_awards, load_players_advanced, load_players_per_game,
    load_teams_advanced, load_teams_per_game,
)
from nba_predict.evaluation.baselines import last_season_baseline
from nba_predict.evaluation.metrics import feature_importance, print_metrics, regression_metrics

MODEL_NAME = "season_outcomes"


# ─── 4A: Win Totals ─────────────────────────────────────────────────────────

def _build_win_totals_dataset() -> pd.DataFrame:
    """Build team-season dataset for win total prediction.

    Features: prior-season team stats (off_rtg, def_rtg, srs, pace, etc.)
              + roster quality proxies from prior-season player data.
    Target:   current-season wins.
    """
    teams_adv = load_teams_advanced()
    teams_pg = load_teams_per_game()

    # Current season targets: wins
    targets = teams_adv[["team", "season", "wins"]].copy()
    targets = targets.rename(columns={"wins": "target_wins"})
    targets["target_wins"] = pd.to_numeric(targets["target_wins"], errors="coerce")

    # Prior-season features: shift everything forward by 1 season
    features = teams_adv[["team", "season", "wins", "losses", "off_rtg", "def_rtg",
                           "net_rtg", "pace", "srs", "sos", "mov", "ts_pct",
                           "orb_pct", "drb_pct"]].copy()
    features["pred_season"] = features["season"] + 1

    rename_map = {c: f"prev_{c}" for c in features.columns
                  if c not in ("team", "season", "pred_season")}
    features = features.rename(columns=rename_map)

    # Add prior-season per-game shooting
    pg_cols = ["team", "season", "fg_pct", "fg3_pct", "ft_pct", "ast", "tov", "pts"]
    teams_pg_sub = teams_pg[pg_cols].copy()
    teams_pg_sub["pred_season"] = teams_pg_sub["season"] + 1
    pg_rename = {c: f"prev_{c}" for c in pg_cols if c not in ("team", "season", "pred_season")}
    teams_pg_sub = teams_pg_sub.rename(columns=pg_rename)

    features = features.merge(
        teams_pg_sub.drop(columns=["season"]),
        on=["team", "pred_season"],
        how="left",
    )

    # Add roster quality signals from player data
    players_adv = load_players_advanced()
    roster_qual = _aggregate_roster_for_season(players_adv)
    roster_qual["pred_season"] = roster_qual["season"] + 1

    features = features.merge(
        roster_qual.drop(columns=["season"]),
        on=["team", "pred_season"],
        how="left",
    )

    # Join features to targets
    df = targets.merge(
        features.drop(columns=["season"]),
        left_on=["team", "season"],
        right_on=["team", "pred_season"],
        how="inner",
    )

    df = df.dropna(subset=["target_wins"])

    # Regression-to-mean feature: how far from .500 (41 wins) was the prior season?
    # Extreme teams tend to regress — this encodes that prior knowledge.
    if "prev_wins" in df.columns:
        df["prev_wins_deviation"] = pd.to_numeric(df["prev_wins"], errors="coerce") - 41.0

    return df


def _aggregate_roster_for_season(players_adv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player advanced stats into team-level roster quality."""
    # Filter to meaningful players
    valid = players_adv[(players_adv["g"] >= 20) & (players_adv["mp"] >= 200)].copy()

    def _team_agg(group):
        top5 = group.nlargest(5, "mp")
        return pd.Series({
            "roster_total_vorp": group["vorp"].sum(),
            "roster_total_ws": group["ws"].sum(),
            "roster_top5_bpm": top5["bpm"].mean(),
            "roster_avg_age": group["age"].mean(),
            "roster_size_qual": len(group),  # players meeting min thresholds
        })

    result = valid.groupby(["team", "season"]).apply(_team_agg, include_groups=False).reset_index()
    return result


def _get_win_totals_features() -> list[str]:
    """Feature columns for win totals model."""
    return [
        "prev_wins", "prev_losses", "prev_off_rtg", "prev_def_rtg",
        "prev_net_rtg", "prev_pace", "prev_srs", "prev_sos", "prev_mov",
        "prev_ts_pct", "prev_orb_pct", "prev_drb_pct",
        "prev_fg_pct", "prev_fg3_pct", "prev_ft_pct", "prev_ast", "prev_tov", "prev_pts",
        "roster_total_vorp", "roster_total_ws", "roster_top5_bpm",
        "roster_avg_age", "roster_size_qual",
        # Regression-to-mean signal
        "prev_wins_deviation",
    ]


# ─── 4B: Championship Odds ──────────────────────────────────────────────────

def _compute_championship_odds(predictions: pd.DataFrame) -> pd.DataFrame:
    """Derive championship probability from predicted wins and SRS.

    Uses a simple calibration: historically, teams with more wins and
    higher SRS win championships more often. We use a softmax-like
    scoring to convert predicted wins into probabilities.
    """
    df = predictions.copy()

    # Composite score: predicted wins + SRS contribution
    srs_col = "prev_srs" if "prev_srs" in df.columns else None
    df["composite"] = df["pred_wins"]
    if srs_col and srs_col in df.columns:
        df["composite"] = df["composite"] + df[srs_col].fillna(0) * 0.5

    # Softmax to convert into probabilities per season
    def _season_probs(group):
        scores = group["composite"].values
        # Temperature-scaled softmax — higher temperature = more spread out
        exp_scores = np.exp((scores - scores.max()) / 3.0)
        group["championship_prob"] = exp_scores / exp_scores.sum()
        group["championship_rank"] = group["composite"].rank(ascending=False).astype(int)
        return group

    df = df.groupby("season").apply(_season_probs, include_groups=False)
    df = df.reset_index(level=0)
    return df.sort_values(["season", "championship_rank"])


# ─── 4C: MVP Race ───────────────────────────────────────────────────────────

def _build_mvp_dataset() -> pd.DataFrame:
    """Build player-season dataset for MVP award_share prediction.

    Features: individual advanced stats + team wins (proxy for team success).
    Target:   award_share (0-1) from MVP voting.
    """
    awards = load_awards()
    # Filter to MVP-type awards
    if "award_type" in awards.columns:
        mvp = awards[awards["award_type"].astype(str).str.contains("mvp", case=False, na=False)].copy()
    else:
        mvp = awards.copy()

    mvp = mvp[mvp["award_share"].notna() & (mvp["award_share"] > 0)]
    mvp = mvp[["player", "season", "award_share"]].copy()
    mvp = mvp.rename(columns={"award_share": "target_award_share"})

    # Player stats: merge per-game + advanced
    per_game = load_players_per_game()
    advanced = load_players_advanced()

    player_stats = per_game.merge(
        advanced[["player", "team", "season", "per", "bpm", "vorp", "ws", "ws_per_48", "usg_pct"]],
        on=["player", "team", "season"],
        how="inner",
    )

    # Add team wins for context (MVPs almost always come from winning teams)
    teams_adv = load_teams_advanced()
    team_wins = teams_adv[["team", "season", "wins", "srs"]].copy()
    team_wins = team_wins.rename(columns={"wins": "team_wins", "srs": "team_srs"})
    player_stats = player_stats.merge(team_wins, on=["team", "season"], how="left")

    # Join MVP targets — left join so we get all qualifying players
    # Non-MVP-voted players get target = 0
    qualified = player_stats[(player_stats["g"] >= 50) & (player_stats["mp_per_g"] >= 30)].copy()
    df = qualified.merge(mvp, on=["player", "season"], how="left")
    df["target_award_share"] = df["target_award_share"].fillna(0.0)

    return df


def _get_mvp_features() -> list[str]:
    """Feature columns for MVP prediction."""
    return [
        "pts_per_g", "ast_per_g", "trb_per_g", "stl_per_g", "blk_per_g",
        "mp_per_g", "fg_pct", "fg3_pct", "ft_pct",
        "per", "bpm", "vorp", "ws", "ws_per_48", "usg_pct",
        "age", "g", "team_wins", "team_srs",
    ]


# ─── Training Orchestrator ──────────────────────────────────────────────────

def train() -> dict:
    """Train all season outcome models. Returns evaluation results."""
    results = {}

    # ─── 4A: Win Totals (shrinkage + Ridge residual correction) ──────────
    print("\n" + "=" * 60)
    print("  MODEL 4A: WIN TOTALS (shrinkage)")
    print("=" * 60)

    wt_df = _build_win_totals_dataset()
    wt_features = [c for c in _get_win_totals_features() if c in wt_df.columns]

    train_wt = wt_df[wt_df["season"].isin(TRAIN_SEASONS)]
    val_wt = wt_df[wt_df["season"].isin(VAL_SEASONS)]
    test_wt = wt_df[wt_df["season"].isin(TEST_SEASONS)]

    print(f"  Team-seasons: {len(wt_df):,}")
    print(f"  Features: {len(wt_features)}")
    print(f"  Train: {len(train_wt):,} | Val: {len(val_wt):,} | Test: {len(test_wt):,}")

    X_train_wt = train_wt[wt_features].astype(float)
    y_train_wt = train_wt["target_wins"].astype(float)
    X_val_wt = val_wt[wt_features].astype(float)
    y_val_wt = val_wt["target_wins"].astype(float)
    X_test_wt = test_wt[wt_features].astype(float)
    y_test_wt = test_wt["target_wins"].astype(float)

    # Shrinkage baseline (exp001/exp025/exp028):
    # pred = (1-s)*prev_wins + s*league_mean
    # Per-season league mean handles shortened seasons (2020=72, 2012=66).
    prev_wins_train = X_train_wt["prev_wins"].astype(float).values
    prev_wins_val = X_val_wt["prev_wins"].astype(float).values
    prev_wins_test = X_test_wt["prev_wins"].astype(float).values

    def _per_season_mean(df, col="prev_wins"):
        return df.groupby("season")[col].transform("mean").astype(float).values

    shrink_target_train = _per_season_mean(train_wt)
    shrink_target_val = _per_season_mean(val_wt)
    shrink_target_test = _per_season_mean(test_wt)

    # Optimize shrinkage on val only (exp028: val-only is simpler + marginally better)
    best_s, best_val_mae = 0.0, float("inf")
    for s in np.arange(0.0, 0.60, 0.01):
        pred = (1 - s) * prev_wins_val + s * shrink_target_val
        mae = float(np.mean(np.abs(y_val_wt.values - pred)))
        if mae < best_val_mae:
            best_val_mae, best_s = mae, s

    shrink_pred_train = (1 - best_s) * prev_wins_train + best_s * shrink_target_train
    shrink_pred_val = (1 - best_s) * prev_wins_val + best_s * shrink_target_val
    shrink_pred_test = (1 - best_s) * prev_wins_test + best_s * shrink_target_test
    print(f"  Shrinkage: s={best_s:.2f}")

    # Ridge residual correction (exp027): learn corrections on top of shrinkage
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_wt.fillna(0))
    X_val_sc = scaler.transform(X_val_wt.fillna(0))
    X_test_sc = scaler.transform(X_test_wt.fillna(0))

    resid_train = y_train_wt.values - shrink_pred_train

    best_alpha, best_corr_val_mae = 100.0, float("inf")
    for alpha in [10, 50, 100, 500, 1000, 5000, 10000]:
        r = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        r.fit(X_train_sc, resid_train)
        correction = r.predict(X_val_sc)
        combo = shrink_pred_val + correction
        vm = float(np.mean(np.abs(y_val_wt.values - combo)))
        if vm < best_corr_val_mae:
            best_corr_val_mae, best_alpha = vm, alpha

    # Only use correction if it beats pure shrinkage
    no_corr_val_mae = float(np.mean(np.abs(y_val_wt.values - shrink_pred_val)))

    if no_corr_val_mae <= best_corr_val_mae:
        wt_pred = shrink_pred_test
        ridge = None
        print(f"  Pure shrinkage (Ridge correction didn't help on val)")
    else:
        ridge = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
        X_tv = np.vstack([X_train_sc, X_val_sc])
        resid_tv = np.concatenate([resid_train, y_val_wt.values - shrink_pred_val])
        ridge.fit(X_tv, resid_tv)
        wt_pred = shrink_pred_test + ridge.predict(X_test_sc)
        print(f"  Shrinkage + Ridge correction (alpha={best_alpha})")

    # We still need an XGBoost model for feature importance display
    wt_xgb_params = {
        **XGBOOST_REGRESSOR_PARAMS,
        "max_depth": 3, "reg_alpha": 2.0, "reg_lambda": 5.0, "learning_rate": 0.03,
    }
    wt_model = XGBRegressor(**wt_xgb_params)
    wt_model.fit(X_train_wt, y_train_wt, eval_set=[(X_val_wt, y_val_wt)], verbose=False)

    wt_metrics = regression_metrics(y_test_wt.values, wt_pred)
    print_metrics("Win Totals (shrinkage) — Test Set", wt_metrics)

    wt_bl = last_season_baseline(y_test_wt.values, test_wt["prev_wins"].astype(float).values)
    print(f"  Baseline (last season wins): MAE={wt_bl['mae']:.2f}")

    wt_improvement = ((wt_bl["mae"] - wt_metrics["mae"]) / wt_bl["mae"]) * 100
    print(f"  Improvement: {wt_improvement:+.1f}%")

    fi_wt = feature_importance(wt_model, wt_features, top_n=10)
    print("\n  Top 10 Features:")
    for _, row in fi_wt.iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    results["win_totals"] = {
        "model": wt_model,
        "test_metrics": wt_metrics,
        "baseline": wt_bl,
        "improvement_pct": wt_improvement,
        "feature_importance": fi_wt,
    }

    # ─── 4B: Championship Odds ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL 4B: CHAMPIONSHIP ODDS (derived from win totals)")
    print("=" * 60)

    test_wt_copy = test_wt.copy()
    test_wt_copy["pred_wins"] = wt_pred
    champ_df = _compute_championship_odds(test_wt_copy)

    for season in sorted(test_wt_copy["season"].unique()):
        season_df = champ_df[champ_df["season"] == season].head(5)
        print(f"\n  {season} Season — Top 5 Championship Contenders:")
        for _, row in season_df.iterrows():
            print(f"    {row['team']:4s}  Pred Wins: {row['pred_wins']:.1f}  "
                  f"Prob: {row['championship_prob']:.1%}  Rank: {row['championship_rank']}")

    results["championship_odds"] = {
        "sample_predictions": champ_df,
    }

    # ─── 4C: MVP Race (XGB+Ridge blend, exp031) ─────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL 4C: MVP RACE (XGB+Ridge blend)")
    print("=" * 60)

    mvp_df = _build_mvp_dataset()
    mvp_features = [c for c in _get_mvp_features() if c in mvp_df.columns]

    train_mvp = mvp_df[mvp_df["season"].isin(TRAIN_SEASONS)]
    val_mvp = mvp_df[mvp_df["season"].isin(VAL_SEASONS)]
    test_mvp = mvp_df[mvp_df["season"].isin(TEST_SEASONS)]

    print(f"  Qualifying player-seasons: {len(mvp_df):,}")
    print(f"  Features: {len(mvp_features)}")
    print(f"  Train: {len(train_mvp):,} | Val: {len(val_mvp):,} | Test: {len(test_mvp):,}")

    X_train_mvp = train_mvp[mvp_features].astype(float)
    y_train_mvp = train_mvp["target_award_share"].astype(float)
    X_val_mvp = val_mvp[mvp_features].astype(float)
    y_val_mvp = val_mvp["target_award_share"].astype(float)
    X_test_mvp = test_mvp[mvp_features].astype(float)
    y_test_mvp = test_mvp["target_award_share"].astype(float)

    mvp_model = XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
    mvp_model.fit(X_train_mvp, y_train_mvp, eval_set=[(X_val_mvp, y_val_mvp)], verbose=False)

    # XGB+Ridge blend (exp031): Ridge captures linear stat→vote relationships
    mvp_scaler = StandardScaler()
    X_train_mvp_sc = mvp_scaler.fit_transform(X_train_mvp.fillna(0))
    X_val_mvp_sc = mvp_scaler.transform(X_val_mvp.fillna(0))
    X_test_mvp_sc = mvp_scaler.transform(X_test_mvp.fillna(0))

    xgb_val_pred = mvp_model.predict(X_val_mvp)
    xgb_test_pred = mvp_model.predict(X_test_mvp)
    voted_val = y_val_mvp.values > 0

    best_val_rho, best_mvp_w, best_mvp_alpha = -1.0, 1.0, 100.0
    for alpha in [0.1, 0.5, 1, 5, 10, 50, 100, 200, 500]:
        r = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        r.fit(X_train_mvp_sc, y_train_mvp)
        r_val = r.predict(X_val_mvp_sc)
        for w in np.arange(0.50, 1.01, 0.01):
            blend = w * xgb_val_pred + (1 - w) * r_val
            if voted_val.sum() >= 5:
                rv, _ = spearmanr(y_val_mvp.values[voted_val], blend[voted_val])
                if rv > best_val_rho:
                    best_val_rho, best_mvp_w, best_mvp_alpha = rv, w, alpha

    if best_mvp_w < 1.0:
        mvp_ridge = Ridge(alpha=best_mvp_alpha, random_state=RANDOM_SEED)
        mvp_ridge.fit(X_train_mvp_sc, y_train_mvp)
        mvp_pred = best_mvp_w * xgb_test_pred + (1 - best_mvp_w) * mvp_ridge.predict(X_test_mvp_sc)
        print(f"  MVP blend: {best_mvp_w:.2f} XGB + {1-best_mvp_w:.2f} Ridge (alpha={best_mvp_alpha})")
    else:
        mvp_pred = xgb_test_pred
        mvp_ridge = None
        print(f"  MVP: pure XGB (Ridge blend didn't help on val)")

    mvp_metrics = regression_metrics(y_test_mvp.values, mvp_pred)
    print_metrics("MVP Award Share — Test Set", mvp_metrics)

    voted_mask = y_test_mvp.values > 0
    if voted_mask.sum() >= 5:
        rho, p_val = spearmanr(y_test_mvp.values[voted_mask], mvp_pred[voted_mask])
        print(f"  Spearman rank correlation (voted players): {rho:.4f} (p={p_val:.4f})")
    else:
        rho, p_val = np.nan, np.nan
        print("  Not enough voted players in test set for rank correlation")

    test_mvp_copy = test_mvp.copy()
    test_mvp_copy["pred_award_share"] = mvp_pred
    for season in sorted(test_mvp_copy["season"].unique()):
        season_df = test_mvp_copy[test_mvp_copy["season"] == season].nlargest(5, "pred_award_share")
        print(f"\n  {season} Season — Top 5 Predicted MVP Candidates:")
        for _, row in season_df.iterrows():
            actual = row["target_award_share"]
            pred = row["pred_award_share"]
            print(f"    {row['player']:25s}  Pred: {pred:.3f}  Actual: {actual:.3f}")

    fi_mvp = feature_importance(mvp_model, mvp_features, top_n=10)
    print("\n  Top 10 MVP Features:")
    for _, row in fi_mvp.iterrows():
        print(f"    {row['feature']:30s} {row['importance']:.4f}")

    results["mvp_race"] = {
        "model": mvp_model,
        "test_metrics": mvp_metrics,
        "spearman_rho": rho,
        "spearman_p": p_val,
        "feature_importance": fi_mvp,
    }

    # ─── Save ────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    save_data = {
        "win_totals_model": wt_model,
        "win_totals_ridge": ridge,
        "win_totals_scaler": scaler,
        "win_totals_features": wt_features,
        "win_totals_shrinkage": best_s,
        "mvp_model": mvp_model,
        "mvp_features": mvp_features,
        "mvp_ridge": mvp_ridge if best_mvp_w < 1.0 else None,
        "mvp_scaler": mvp_scaler if best_mvp_w < 1.0 else None,
        "mvp_blend_weight": best_mvp_w,
    }
    joblib.dump(save_data, model_path)
    print(f"\n  Models saved: {model_path}")

    return {
        "model_name": MODEL_NAME,
        "results": results,
    }
