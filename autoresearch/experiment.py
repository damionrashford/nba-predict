"""
AutoResearch Experiment — THE ONLY EDITABLE FILE.

This file contains all model training logic for one experiment run.
It imports data/features from nba_predict (sacred, read-only) and
defines its own model architectures and training procedures.

Modify this file freely to test hypotheses. The evaluator will call
run_experiment() and compute NBA_CORE from the returned results.

Returns dict with keys:
    game_winner:  {accuracy, auc_roc}
    point_spread: {mae, rmse, derived_accuracy}
    player_pts:   {mae}
    player_ast:   {mae}
    player_reb:   {mae}
    win_totals:   {mae}
    mvp_race:     {mae, spearman}
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predict.config import (
    RANDOM_SEED,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    XGBOOST_CLASSIFIER_PARAMS,
    XGBOOST_REGRESSOR_PARAMS,
)
from nba_predict.data.loader import (
    load_awards,
    load_players_advanced,
    load_players_per_game,
    load_teams_advanced,
    load_teams_per_game,
)
from nba_predict.evaluation.metrics import (
    classification_metrics,
    regression_metrics,
)
from nba_predict.features.matchup_features import (
    build_matchup_dataset,
    get_feature_columns,
)

# New data sources (injury + tracking)
from autoresearch.injury_features import build_injury_features, join_injury_features
from autoresearch.tracking_features import build_tracking_features, join_tracking_features

# Experiment outputs go here (NOT production outputs/)
EXPERIMENT_MODELS_DIR = Path(__file__).parent / "outputs" / "models"
EXPERIMENT_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _build_enriched_matchup_dataset() -> pd.DataFrame:
    """Build matchup dataset enriched with injury + tracking features."""
    df = build_matchup_dataset()

    # Add injury features (seasons 2022-2026, NaN for earlier)
    injury_feats = build_injury_features()
    df = join_injury_features(df, injury_feats)

    # Add tracking features (seasons 2015-2027 as prior-season, NaN for earlier)
    tracking_feats = build_tracking_features()
    df = join_tracking_features(df, tracking_feats)

    return df


def run_experiment() -> dict:
    """Run all models and return standardized results dict."""
    results = {}

    gw = _train_game_winner()
    results["game_winner"] = gw

    ps = _train_point_spread()
    results["point_spread"] = ps

    pp = _train_player_performance()
    results["player_pts"] = pp["pts"]
    results["player_ast"] = pp["ast"]
    results["player_reb"] = pp["reb"]

    so = _train_season_outcomes()
    results["win_totals"] = so["win_totals"]
    results["mvp_race"] = so["mvp_race"]

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: GAME WINNER (Binary Classification)
# ═══════════════════════════════════════════════════════════════════════════

def _train_game_winner() -> dict:
    """Train game winner model. Returns {accuracy, auc_roc}."""
    print("\n  [1/4] Game Winner...")

    df = build_matchup_dataset()
    feature_cols = get_feature_columns(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["home_win"].astype(int)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["home_win"].astype(int)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["home_win"].astype(int)

    # Train XGBoost with early stopping
    model = XGBClassifier(**XGBOOST_CLASSIFIER_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Skip isotonic calibration — XGBoost's logistic output is already well-calibrated
    # and isotonic overfits to val distribution (hurts test accuracy by ~0.002)
    raw_prob = model.predict_proba(X_test)[:, 1]
    y_prob = raw_prob
    y_pred = (y_prob > 0.5).astype(int)

    metrics = classification_metrics(y_test.values, y_pred, y_prob)

    # Save to experiment dir
    joblib.dump(
        {"model": model, "calibrator": None, "feature_cols": feature_cols},
        EXPERIMENT_MODELS_DIR / "game_winner.joblib",
    )

    print(f"    acc={metrics['accuracy']:.4f}  auc={metrics.get('auc_roc', 0):.4f}")
    return {"accuracy": metrics["accuracy"], "auc_roc": metrics.get("auc_roc", 0)}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: POINT SPREAD (Regression)
# ═══════════════════════════════════════════════════════════════════════════

def _train_point_spread() -> dict:
    """Train point spread model. Returns {mae, rmse, derived_accuracy}."""
    print("\n  [2/4] Point Spread...")

    df = build_matchup_dataset()
    feature_cols = get_feature_columns(df)

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["margin"].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["margin"].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["margin"].astype(float)

    # Huber loss XGBoost with tuned hyperparams + Ridge blend
    huber_params = {
        **XGBOOST_REGRESSOR_PARAMS,
        "objective": "reg:pseudohubererror",
        "max_depth": 5,
        "n_estimators": 800,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "reg_lambda": 2.0,
        "colsample_bytree": 0.7,
    }
    model = XGBRegressor(**huber_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Ridge on scaled features
    ps_scaler = StandardScaler()
    X_train_sc = ps_scaler.fit_transform(X_train.fillna(0))
    X_val_sc = ps_scaler.transform(X_val.fillna(0))
    X_test_sc = ps_scaler.transform(X_test.fillna(0))
    # Sweep Ridge alpha and blend ratio jointly on val
    xgb_val = model.predict(X_val)
    best_blend, best_blend_mae, best_alpha = 1.0, float("inf"), 100.0
    for alpha in [10, 50, 100, 200, 500, 1000]:
        r = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        r.fit(X_train_sc, y_train)
        r_val = r.predict(X_val_sc)
        for w in np.arange(0.5, 1.01, 0.05):
            bp = w * xgb_val + (1 - w) * r_val
            bm = float(np.mean(np.abs(y_val.values - bp)))
            if bm < best_blend_mae:
                best_blend_mae, best_blend, best_alpha = bm, w, alpha

    ps_ridge = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
    ps_ridge.fit(X_train_sc, y_train)
    print(f"    ridge_alpha={best_alpha}")

    xgb_test = model.predict(X_test)
    ridge_test = ps_ridge.predict(X_test_sc)
    y_pred = best_blend * xgb_test + (1 - best_blend) * ridge_test
    print(f"    blend={best_blend:.2f} val_mae={best_blend_mae:.2f}")
    metrics = regression_metrics(y_test.values, y_pred)

    # Derived accuracy: does predicted spread sign match actual winner?
    pred_winner = (y_pred > 0).astype(int)
    actual_winner = (y_test.values > 0).astype(int)
    derived_acc = float((pred_winner == actual_winner).mean())

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        EXPERIMENT_MODELS_DIR / "point_spread.joblib",
    )

    print(f"    mae={metrics['mae']:.2f}  derived_acc={derived_acc:.4f}")
    return {"mae": metrics["mae"], "rmse": metrics["rmse"], "derived_accuracy": derived_acc}


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: PLAYER PERFORMANCE (Multi-target Regression)
# ═══════════════════════════════════════════════════════════════════════════

MIN_GAMES = 20
MIN_MINUTES = 10.0


def _build_player_dataset() -> pd.DataFrame:
    """Build player-season dataset. Features = season N-1, targets = season N."""
    per_game = load_players_per_game()
    advanced = load_players_advanced()
    teams_adv = load_teams_advanced()

    per_game = per_game[(per_game["g"] >= MIN_GAMES) & (per_game["mp_per_g"] >= MIN_MINUTES)]
    advanced = advanced[advanced["g"] >= MIN_GAMES]

    # Position encoding
    pos_map = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    if "pos" in per_game.columns:
        per_game["pos_encoded"] = (
            per_game["pos"].astype(str).str.split("-").str[0].map(pos_map).fillna(3)
        )
    else:
        per_game["pos_encoded"] = 3

    # Games-started ratio
    gs_col = "gs" if "gs" in per_game.columns else "games_started"
    if gs_col in per_game.columns and "g" in per_game.columns:
        per_game["gs_ratio"] = (
            pd.to_numeric(per_game[gs_col], errors="coerce").fillna(0)
            / per_game["g"].clip(lower=1)
        )
    else:
        per_game["gs_ratio"] = 0.5

    # Merge per_game + advanced
    player_stats = per_game.merge(
        advanced[["player", "team", "season", "per", "ts_pct", "bpm", "obpm", "dbpm",
                  "vorp", "ws", "ws_per_48", "usg_pct", "mp"]],
        on=["player", "team", "season"],
        how="inner",
        suffixes=("", "_adv"),
    )

    # Self-join: season N-1 features -> season N targets
    features = player_stats.copy()
    features["next_season"] = features["season"] + 1

    targets = player_stats[["player", "season", "team", "pts_per_g", "ast_per_g",
                             "trb_per_g", "g", "mp_per_g"]].copy()
    targets = targets.rename(columns={
        "pts_per_g": "target_pts", "ast_per_g": "target_ast",
        "trb_per_g": "target_reb", "g": "target_g",
        "mp_per_g": "target_mp", "team": "target_team",
    })

    df = features.merge(
        targets, left_on=["player", "next_season"], right_on=["player", "season"],
        how="inner", suffixes=("", "_target"),
    )

    df["season"] = df["season_target"]
    df["feature_season"] = df["season"] - 1

    # Team context from prior season
    team_ctx = teams_adv[["team", "season", "off_rtg", "def_rtg", "pace", "srs"]].copy()
    team_ctx = team_ctx.rename(columns={
        "off_rtg": "team_off_rtg", "def_rtg": "team_def_rtg",
        "pace": "team_pace", "srs": "team_srs",
    })
    df = df.merge(
        team_ctx, left_on=["team", "feature_season"], right_on=["team", "season"],
        how="left", suffixes=("", "_team"),
    )

    df["age_next"] = df["age"] + 1

    season_counts = features.groupby("player")["season"].nunique().reset_index()
    season_counts.columns = ["player", "career_seasons"]
    df = df.merge(season_counts, on="player", how="left")

    # Interaction features
    df["pos_x_orb"] = df["pos_encoded"] * df["orb_per_g"]
    df["pos_x_drb"] = df["pos_encoded"] * df["drb_per_g"]
    df["pos_x_trb"] = df["pos_encoded"] * df["trb_per_g"]
    df["mp_x_trb"] = df["mp_per_g"] * df["trb_per_g"]
    df["usg_x_pts"] = df["usg_pct"] * df["pts_per_g"]
    df["mp_x_pts"] = df["mp_per_g"] * df["pts_per_g"]
    df["ts_x_usg"] = df["ts_pct"] * df["usg_pct"]

    return df


def _get_player_feature_columns() -> list[str]:
    return [
        "pts_per_g", "ast_per_g", "trb_per_g", "stl_per_g", "blk_per_g",
        "mp_per_g", "fg_pct", "fg3_pct", "ft_pct", "tov_per_g",
        "orb_per_g", "drb_per_g",
        "per", "ts_pct", "bpm", "obpm", "dbpm", "vorp", "ws", "ws_per_48", "usg_pct",
        "age", "age_next", "career_seasons", "pos_encoded", "gs_ratio", "g",
        "team_off_rtg", "team_def_rtg", "team_pace", "team_srs",
        "pos_x_orb", "pos_x_drb", "pos_x_trb", "mp_x_trb",
        "usg_x_pts", "mp_x_pts", "ts_x_usg",
    ]


def _train_player_performance() -> dict:
    """Train player PTS/AST/REB models. Returns dict of per-target results."""
    print("\n  [3/4] Player Performance...")

    df = _build_player_dataset()
    feature_cols = [c for c in _get_player_feature_columns() if c in df.columns]

    targets = {"pts": "target_pts", "ast": "target_ast", "reb": "target_reb"}

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"].isin(VAL_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]

    results = {}
    models = {}

    # Per-target hyperparameters
    target_params = {
        "pts": {**XGBOOST_REGRESSOR_PARAMS, "max_depth": 5, "n_estimators": 600,
                "learning_rate": 0.04, "min_child_weight": 3},
        "ast": {**XGBOOST_REGRESSOR_PARAMS, "max_depth": 3, "n_estimators": 800,
                "learning_rate": 0.03, "min_child_weight": 8, "reg_lambda": 3.0,
                "reg_alpha": 0.5},
        "reb": {**XGBOOST_REGRESSOR_PARAMS, "max_depth": 4, "reg_alpha": 1.0,
                "reg_lambda": 3.0, "min_child_weight": 8},
    }

    # Per-target feature exclusions (remove noisy interactions)
    target_exclude = {
        "pts": [],
        "ast": [],
        "reb": ["usg_x_pts", "mp_x_pts", "ts_x_usg"],  # scoring interactions hurt REB
    }

    for target_name, target_col in targets.items():
        y_train = train_df[target_col].astype(float)
        y_val = val_df[target_col].astype(float)
        y_test = test_df[target_col].astype(float)

        # Use per-target feature set
        excl = target_exclude.get(target_name, [])
        t_cols = [c for c in feature_cols if c not in excl]

        params = target_params.get(target_name, XGBOOST_REGRESSOR_PARAMS)
        model = XGBRegressor(**params)
        model.fit(train_df[t_cols].astype(float), y_train,
                  eval_set=[(val_df[t_cols].astype(float), y_val)], verbose=False)
        models[target_name] = model

        y_pred = model.predict(test_df[t_cols].astype(float))
        metrics = regression_metrics(y_test.values, y_pred)

        print(f"    {target_name}: mae={metrics['mae']:.2f}")
        results[target_name] = {"mae": metrics["mae"]}

    joblib.dump(
        {"models": models, "feature_cols": feature_cols, "targets": list(targets.keys())},
        EXPERIMENT_MODELS_DIR / "player_performance.joblib",
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4: SEASON OUTCOMES (Win Totals + MVP Race)
# ═══════════════════════════════════════════════════════════════════════════

def _build_win_totals_dataset() -> pd.DataFrame:
    """Build team-season dataset for win total prediction."""
    teams_adv = load_teams_advanced()
    teams_pg = load_teams_per_game()

    targets = teams_adv[["team", "season", "wins"]].copy()
    targets = targets.rename(columns={"wins": "target_wins"})
    targets["target_wins"] = pd.to_numeric(targets["target_wins"], errors="coerce")

    # Prior-season features
    features = teams_adv[["team", "season", "wins", "losses", "off_rtg", "def_rtg",
                           "net_rtg", "pace", "srs", "sos", "mov", "ts_pct",
                           "orb_pct", "drb_pct"]].copy()
    features["pred_season"] = features["season"] + 1
    rename_map = {c: f"prev_{c}" for c in features.columns
                  if c not in ("team", "season", "pred_season")}
    features = features.rename(columns=rename_map)

    # Prior-season per-game shooting
    pg_cols = ["team", "season", "fg_pct", "fg3_pct", "ft_pct", "ast", "tov", "pts"]
    teams_pg_sub = teams_pg[pg_cols].copy()
    teams_pg_sub["pred_season"] = teams_pg_sub["season"] + 1
    pg_rename = {c: f"prev_{c}" for c in pg_cols if c not in ("team", "season", "pred_season")}
    teams_pg_sub = teams_pg_sub.rename(columns=pg_rename)
    features = features.merge(
        teams_pg_sub.drop(columns=["season"]), on=["team", "pred_season"], how="left",
    )

    # Roster quality
    players_adv = load_players_advanced()
    roster_qual = _aggregate_roster(players_adv)
    roster_qual["pred_season"] = roster_qual["season"] + 1
    features = features.merge(
        roster_qual.drop(columns=["season"]), on=["team", "pred_season"], how="left",
    )

    df = targets.merge(
        features.drop(columns=["season"]),
        left_on=["team", "season"], right_on=["team", "pred_season"], how="inner",
    )
    df = df.dropna(subset=["target_wins"])

    if "prev_wins" in df.columns:
        df["prev_wins_deviation"] = pd.to_numeric(df["prev_wins"], errors="coerce") - 41.0

    return df


def _aggregate_roster(players_adv: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player stats into team-level roster quality."""
    valid = players_adv[(players_adv["g"] >= 20) & (players_adv["mp"] >= 200)].copy()

    def _team_agg(group):
        top5 = group.nlargest(5, "mp")
        return pd.Series({
            "roster_total_vorp": group["vorp"].sum(),
            "roster_total_ws": group["ws"].sum(),
            "roster_top5_bpm": top5["bpm"].mean(),
            "roster_avg_age": group["age"].mean(),
            "roster_size_qual": len(group),
        })

    return valid.groupby(["team", "season"]).apply(
        _team_agg, include_groups=False
    ).reset_index()


def _get_win_totals_features() -> list[str]:
    return [
        "prev_wins", "prev_losses", "prev_off_rtg", "prev_def_rtg",
        "prev_net_rtg", "prev_pace", "prev_srs", "prev_sos", "prev_mov",
        "prev_ts_pct", "prev_orb_pct", "prev_drb_pct",
        "prev_fg_pct", "prev_fg3_pct", "prev_ft_pct", "prev_ast", "prev_tov", "prev_pts",
        "roster_total_vorp", "roster_total_ws", "roster_top5_bpm",
        "roster_avg_age", "roster_size_qual",
        "prev_wins_deviation",
    ]


def _build_mvp_dataset() -> pd.DataFrame:
    """Build player-season dataset for MVP prediction."""
    awards = load_awards()
    if "award_type" in awards.columns:
        mvp = awards[awards["award_type"].astype(str).str.contains("mvp", case=False, na=False)].copy()
    else:
        mvp = awards.copy()

    mvp = mvp[mvp["award_share"].notna() & (mvp["award_share"] > 0)]
    mvp = mvp[["player", "season", "award_share"]].copy()
    mvp = mvp.rename(columns={"award_share": "target_award_share"})

    per_game = load_players_per_game()
    advanced = load_players_advanced()

    player_stats = per_game.merge(
        advanced[["player", "team", "season", "per", "bpm", "vorp", "ws", "ws_per_48", "usg_pct"]],
        on=["player", "team", "season"], how="inner",
    )

    teams_adv = load_teams_advanced()
    team_wins = teams_adv[["team", "season", "wins", "srs"]].copy()
    team_wins = team_wins.rename(columns={"wins": "team_wins", "srs": "team_srs"})
    player_stats = player_stats.merge(team_wins, on=["team", "season"], how="left")

    qualified = player_stats[
        (player_stats["g"] >= 50) & (player_stats["mp_per_g"] >= 30)
    ].copy()

    df = qualified.merge(mvp, on=["player", "season"], how="left")
    df["target_award_share"] = df["target_award_share"].fillna(0.0)

    return df


def _get_mvp_features() -> list[str]:
    return [
        "pts_per_g", "ast_per_g", "trb_per_g", "stl_per_g", "blk_per_g",
        "mp_per_g", "fg_pct", "fg3_pct", "ft_pct",
        "per", "bpm", "vorp", "ws", "ws_per_48", "usg_pct",
        "age", "g", "team_wins", "team_srs",
    ]


def _train_season_outcomes() -> dict:
    """Train win totals + MVP models. Returns dict with both."""
    print("\n  [4/4] Season Outcomes...")

    # ── 4A: Win Totals ───────────────────────────────────────────────────
    wt_df = _build_win_totals_dataset()
    wt_features = [c for c in _get_win_totals_features() if c in wt_df.columns]

    train_wt = wt_df[wt_df["season"].isin(TRAIN_SEASONS)]
    val_wt = wt_df[wt_df["season"].isin(VAL_SEASONS)]
    test_wt = wt_df[wt_df["season"].isin(TEST_SEASONS)]

    X_train_wt = train_wt[wt_features].astype(float)
    y_train_wt = train_wt["target_wins"].astype(float)
    X_val_wt = val_wt[wt_features].astype(float)
    y_val_wt = val_wt["target_wins"].astype(float)
    X_test_wt = test_wt[wt_features].astype(float)
    y_test_wt = test_wt["target_wins"].astype(float)

    # Shrinkage baseline: pred = (1-s)*prev_wins + s*prior_league_mean
    # Use per-season mean of prev_wins as the shrinkage target (handles shortened seasons)
    prev_wins_train = X_train_wt["prev_wins"].astype(float).values
    prev_wins_val = X_val_wt["prev_wins"].astype(float).values
    prev_wins_test = X_test_wt["prev_wins"].astype(float).values

    # Per-season league mean of prev_wins (known at prediction time)
    def _per_season_mean(df, col="prev_wins"):
        means = df.groupby("season")[col].transform("mean").astype(float).values
        return means

    shrink_target_train = _per_season_mean(train_wt)
    shrink_target_val = _per_season_mean(val_wt)
    shrink_target_test = _per_season_mean(test_wt)

    # Optimize shrinkage on val (temporally closest to test)
    best_s, best_val_mae = 0.0, float("inf")
    for s in np.arange(0.0, 0.60, 0.01):
        pred = (1 - s) * prev_wins_val + s * shrink_target_val
        mae = float(np.mean(np.abs(y_val_wt.values - pred)))
        if mae < best_val_mae:
            best_val_mae, best_s = mae, s

    final_s = best_s
    shrink_pred_train = (1 - final_s) * prev_wins_train + final_s * shrink_target_train
    shrink_pred_val = (1 - final_s) * prev_wins_val + final_s * shrink_target_val
    shrink_pred_test = (1 - final_s) * prev_wins_test + final_s * shrink_target_test

    # Fit Ridge on residuals (actual - shrinkage_pred) to learn corrections
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_wt.fillna(0))
    X_val_sc = scaler.transform(X_val_wt.fillna(0))
    X_test_sc = scaler.transform(X_test_wt.fillna(0))

    resid_train = y_train_wt.values - shrink_pred_train

    # Sweep alpha for residual model
    best_alpha, best_blend_val_mae = 100.0, float("inf")
    for alpha in [10, 50, 100, 500, 1000, 5000, 10000]:
        r = Ridge(alpha=alpha, random_state=RANDOM_SEED)
        r.fit(X_train_sc, resid_train)
        correction = r.predict(X_val_sc)
        combo = shrink_pred_val + correction
        vm = float(np.mean(np.abs(y_val_wt.values - combo)))
        if vm < best_blend_val_mae:
            best_blend_val_mae, best_alpha = vm, alpha

    # Also test no correction at all
    no_corr_val_mae = float(np.mean(np.abs(y_val_wt.values - shrink_pred_val)))

    if no_corr_val_mae <= best_blend_val_mae:
        wt_pred = shrink_pred_test
        print(f"    win_totals: pure_shrinkage s={final_s:.2f}")
    else:
        r = Ridge(alpha=best_alpha, random_state=RANDOM_SEED)
        X_tv = np.vstack([X_train_sc, X_val_sc])
        resid_tv = np.concatenate([resid_train, y_val_wt.values - shrink_pred_val])
        r.fit(X_tv, resid_tv)
        wt_pred = shrink_pred_test + r.predict(X_test_sc)
        print(f"    win_totals: shrinkage+ridge s={final_s:.2f} alpha={best_alpha}")

    # Save artifacts
    ridge = Ridge(alpha=100.0, random_state=RANDOM_SEED)
    ridge.fit(X_train_sc, y_train_wt)
    wt_model = ridge

    wt_metrics = regression_metrics(y_test_wt.values, wt_pred)
    print(f"    win_totals: mae={wt_metrics['mae']:.2f}")

    # ── 4C: MVP Race ─────────────────────────────────────────────────────
    mvp_df = _build_mvp_dataset()
    mvp_features = [c for c in _get_mvp_features() if c in mvp_df.columns]

    train_mvp = mvp_df[mvp_df["season"].isin(TRAIN_SEASONS)]
    val_mvp = mvp_df[mvp_df["season"].isin(VAL_SEASONS)]
    test_mvp = mvp_df[mvp_df["season"].isin(TEST_SEASONS)]

    X_train_mvp = train_mvp[mvp_features].astype(float)
    y_train_mvp = train_mvp["target_award_share"].astype(float)
    X_val_mvp = val_mvp[mvp_features].astype(float)
    y_val_mvp = val_mvp["target_award_share"].astype(float)
    X_test_mvp = test_mvp[mvp_features].astype(float)
    y_test_mvp = test_mvp["target_award_share"].astype(float)

    mvp_model = XGBRegressor(**XGBOOST_REGRESSOR_PARAMS)
    mvp_model.fit(X_train_mvp, y_train_mvp, eval_set=[(X_val_mvp, y_val_mvp)], verbose=False)

    # XGB+Ridge blend: Ridge captures linear stat→vote relationships
    # that XGBoost may overfit on this small dataset
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

    # Build final blend
    if best_mvp_w < 1.0:
        mvp_ridge = Ridge(alpha=best_mvp_alpha, random_state=RANDOM_SEED)
        mvp_ridge.fit(X_train_mvp_sc, y_train_mvp)
        mvp_pred = best_mvp_w * xgb_test_pred + (1 - best_mvp_w) * mvp_ridge.predict(X_test_mvp_sc)
        print(f"    mvp: XGB+Ridge blend w={best_mvp_w:.2f} alpha={best_mvp_alpha:.0f}")
    else:
        mvp_pred = xgb_test_pred
        print(f"    mvp: pure XGB (Ridge blend didn't help on val)")

    mvp_metrics = regression_metrics(y_test_mvp.values, mvp_pred)

    # Spearman rank correlation on voted players
    voted_mask = y_test_mvp.values > 0
    if voted_mask.sum() >= 5:
        rho, _ = spearmanr(y_test_mvp.values[voted_mask], mvp_pred[voted_mask])
    else:
        rho = 0.0

    print(f"    mvp: mae={mvp_metrics['mae']:.4f}  spearman={rho:.4f}")

    # Save
    joblib.dump({
        "win_totals_model": wt_model, "win_totals_ridge": ridge,
        "win_totals_scaler": scaler, "win_totals_features": wt_features,
        "mvp_model": mvp_model, "mvp_features": mvp_features,
    }, EXPERIMENT_MODELS_DIR / "season_outcomes.joblib")

    return {
        "win_totals": {"mae": wt_metrics["mae"]},
        "mvp_race": {"mae": mvp_metrics["mae"], "spearman": float(rho)},
    }
