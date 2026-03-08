# Experiment Ideas (Prioritized)

## Tier 1: Quick Wins (try first, highest expected NBA_CORE impact)

### 1. Win Totals: Pure Ridge (drop XGBoost from blend)
- **Why**: Current 40/60 XGB/Ridge blend gives 9.60 MAE vs 8.80 baseline = WORSE
- **Change**: Replace `_train_season_outcomes()` blend with Ridge-only
- **Expected**: Win totals MAE < 8.80, NBA_CORE += ~0.02-0.04

### 2. Win Totals: Sweep blend ratios
- **Ratios to try**: 20/80, 10/90, 0/100 (more Ridge weight)
- **Why**: XGBoost overfits on small season-level sample (only ~30 teams/year)

### 3. Win Totals: ElasticNet or Lasso
- **Why**: Ridge keeps all features; L1 regularization may help with noisy features
- **Params**: `l1_ratio=0.5` for ElasticNet, default alpha for Lasso

### 4. Win Totals: Tune Ridge alpha
- **Values**: 1.0, 5.0, 10.0, 20.0, 50.0, 100.0
- **Current**: Using sklearn default (alpha=1.0)

### 5. Game Winner: Drop low-importance features
- **Threshold**: importance < 0.005
- **Why**: ~150 matchup features, many are noise. Reducing noise helps generalization.

## Tier 2: Medium Effort (solid expected improvement)

### 6. Win Totals: BayesianRidge
- **Why**: Built-in uncertainty, automatic regularization tuning
- **Import**: `from sklearn.linear_model import BayesianRidge`

### 7. Game Winner: HistGradientBoostingClassifier
- **Why**: Native missing value handling, often better than XGBoost on tabular data
- **Import**: `from sklearn.ensemble import HistGradientBoostingClassifier`
- **Key params**: `max_iter=500, max_depth=5, learning_rate=0.05`

### 8. Point Spread: Huber loss
- **Why**: Robust to outlier blowout games (30+ point margins)
- **XGBoost**: `objective='reg:pseudohubererror'`
- **Alternative**: `from sklearn.linear_model import HuberRegressor`

### 9. Player Models: Per-target hyperparameters
- **Why**: PTS prediction needs different model depth than REB
- **PTS**: deeper trees (max_depth=6), more estimators
- **REB**: shallower trees (max_depth=3), stronger regularization

### 10. Game Winner: Stacking meta-learner
- **Base**: XGBClassifier + LogisticRegression out-of-fold predictions
- **Meta**: Ridge or LogisticRegression on base predictions
- **Import**: `from sklearn.ensemble import StackingClassifier`

## Tier 3: Ambitious (higher risk, higher potential reward)

### 11. Feature Selection: Mutual information
- **Why**: Automated removal of irrelevant features across all models
- **Import**: `from sklearn.feature_selection import mutual_info_classif, SelectKBest`
- **Apply to**: Game winner (classification), then adapt for regression

### 12. Game Winner: Threshold optimization
- **Why**: Default 0.5 threshold may not be optimal for imbalanced home/away wins
- **Method**: Search thresholds 0.45-0.55 on validation set, pick max accuracy

### 13. Point Spread: Quantile regression
- **Why**: Predict median (robust to outliers) instead of mean
- **XGBoost**: `objective='reg:quantileerror', quantile_alpha=0.5`

### 14. Player REB: Interaction features
- **Features**: `pos_encoded * orb_per_g`, `pos_encoded * drb_per_g`, `height * mp_per_g`
- **Why**: Rebounding is highly position-dependent; interactions capture this

### 15. Win Totals: Team continuity feature
- **Feature**: % roster turnover from prior season
- **Data**: Compare `load_rosters(season)` vs `load_rosters(season-1)`
- **Why**: Teams with high continuity regress less to the mean

## Tier 4: Exploratory (low confidence, learn from results)

### 16. Game Winner: CalibratedClassifierCV
- **Method**: isotonic or sigmoid on XGBoost predictions
- **Why**: Better calibrated probabilities may improve AUC even if accuracy stays

### 17. All Models: Feature whitening (StandardScaler)
- **Why**: Ridge/linear models benefit from scaled features
- **Note**: XGBoost doesn't care, but blends and stacking do

### 18. Point Spread: Ensemble XGB + Ridge
- **Why**: If it works for classification (game winner), might work for regression
- **Blend**: 70/30 XGB/Ridge on point spread predictions

### 19. Player Models: Minutes-weighted loss
- **Why**: Starters (30+ min) matter more than bench players (10 min)
- **Method**: `sample_weight=minutes_played` in XGBoost `.fit()`

### 20. MVP Race: Feature engineering
- **Features**: team wins * individual stats interactions
- **Why**: MVP voting heavily weights team success + individual dominance

## Anti-Patterns (things NOT to try)

- **Neural networks**: Not enough data, XGBoost dominates on tabular
- **Deep ensembles (>3 models)**: Overfitting risk on small val set
- **Feature engineering on sacred pipeline**: FORBIDDEN, only experiment.py is editable
- **Changing temporal splits**: IMMUTABLE
- **New package installs**: Only xgboost, sklearn, scipy, numpy, pandas, joblib
