# ============================================================================
# PIPELINE V2 — shared definitions (identical in the tuning and evaluation notebooks)
# ============================================================================
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

N_DAYS     = 1913                       # sales_train_validation.csv only
MAX_LAG    = 42
FIRST_ROW_DAY = MAX_LAG + 1             # row r forecasts day r + 43 (1-based)
H          = 28                         # window length (days)
CUT        = 1745                       # last day used for classes and characteristics
TUNE_STARTS = [1662, 1690, 1718]        # tuning windows: days 1,662-1,745 (before any test day)
EVAL_STARTS = [1746, 1774, 1802, 1830, 1858]   # 5 test windows: days 1,746-1,885
HOLD_START  = 1886                      # holdout window: days 1,886-1,913
MODELS = ['LGBM', 'XGBOOST', 'CATBOOST', 'HGBR', 'RANDOM_FOREST',
          'ELASTIC_NET', 'EXTRA_TREES', 'GRADIENT_BOOST', 'RIDGE', 'LASSO']
LINEAR = {'RIDGE', 'LASSO', 'ELASTIC_NET'}
N_TRIALS = {'RIDGE': 30, 'LASSO': 30, 'ELASTIC_NET': 30, 'RANDOM_FOREST': 30, 'EXTRA_TREES': 30,
            'LGBM': 50, 'XGBOOST': 50, 'CATBOOST': 50, 'HGBR': 50, 'GRADIENT_BOOST': 50}

def build_features(series):
    """Row for target day s uses only days before s. Lags 7..42 fall on the SAME weekday as s.
    Rolling mean/std (population std, ddof=0) over the w days before s. Vectorised with cumulative sums."""
    x = np.asarray(series, float)
    s_idx = np.arange(MAX_LAG, len(x))
    c1 = np.r_[0.0, np.cumsum(x)]; c2 = np.r_[0.0, np.cumsum(x * x)]
    cols = [x[s_idx - k] for k in (7, 14, 28, 35, 42)]
    def mean_w(w): return (c1[s_idx] - c1[s_idx - w]) / w
    def std_w(w):
        m = mean_w(w); v = (c2[s_idx] - c2[s_idx - w]) / w - m * m
        return np.sqrt(np.clip(v, 0, None))
    cols += [mean_w(w) for w in (7, 14, 28)] + [std_w(w) for w in (7, 28)]
    return np.column_stack(cols), x[s_idx]

def first_sale_row(y):
    """First row whose target is a sale (the product's history starts at its first sale, as in the M5 RMSSE)."""
    nz = np.flatnonzero(y > 0)
    return int(nz[0]) if len(nz) else len(y)

def window_rows(start_day):
    """Training rows = all target days before start_day; test rows = the 28 days from start_day."""
    a = start_day - FIRST_ROW_DAY
    return a, a + H                      # train = rows[:a], test = rows[a:a+H]

def rmsse(y_test, pred, y_train):
    """RMSSE with the M5 scale: mean squared one-step change of the training targets from the first sale onward."""
    t = y_train[first_sale_row(y_train):]
    if len(t) < 2: return np.nan
    scale = np.mean(np.diff(t) ** 2)
    return np.nan if scale == 0 else float(np.sqrt(np.mean((y_test - pred) ** 2) / scale))

def window_ok(y, start_day, min_rows=50):
    """A window is evaluable if the product has >= min_rows training rows after its first sale and a non-zero scale."""
    a, _ = window_rows(start_day)
    t = y[first_sale_row(y[:a]):a]
    return len(t) >= min_rows and np.any(np.diff(t) != 0)

def build_model(name, p, n_jobs=-1):
    p = dict(p)
    if name == 'LGBM':          return lgb.LGBMRegressor(objective='regression', verbosity=-1, n_jobs=n_jobs, random_state=42, n_estimators=200, subsample_freq=1, **p)
    if name == 'XGBOOST':       return XGBRegressor(objective='reg:squarederror', verbosity=0, n_jobs=n_jobs, random_state=42, n_estimators=200, **p)
    if name == 'CATBOOST':      return CatBoostRegressor(verbose=False, random_seed=42, iterations=200, thread_count=n_jobs, allow_writing_files=False, **p)
    if name == 'HGBR':          return HistGradientBoostingRegressor(random_state=42, max_iter=200, **p)
    if name == 'RANDOM_FOREST': return RandomForestRegressor(random_state=42, n_jobs=n_jobs, **p)
    if name == 'EXTRA_TREES':   return ExtraTreesRegressor(random_state=42, n_jobs=n_jobs, **p)
    if name == 'GRADIENT_BOOST':return GradientBoostingRegressor(random_state=42, n_estimators=200, **p)
    if name == 'ELASTIC_NET':   return ElasticNet(max_iter=10000, random_state=42, **p)
    if name == 'LASSO':         return Lasso(max_iter=10000, random_state=42, **p)
    if name == 'RIDGE':         return Ridge(**p)

def fit_predict(name, p, X, y, start_day, n_jobs=-1):
    """Train on the product's rows from its first sale up to the day before start_day; forecast the 28-day window."""
    a, b = window_rows(start_day)
    f = first_sale_row(y[:a])
    Xtr, ytr, Xte, yte = X[f:a], y[f:a], X[a:b], y[a:b]
    if name in LINEAR:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    m = build_model(name, p, n_jobs); m.fit(Xtr, ytr)
    pred = np.clip(m.predict(Xte), 0, None)
    return yte, pred, ytr

def search_space(name, trial):          # identical ranges to the July tuning (cells 85-105)
    s = trial.suggest_float; i = trial.suggest_int
    if name == 'LASSO':  return {'alpha': s('alpha', 1e-4, 100.0, log=True)}
    if name == 'RIDGE':  return {'alpha': s('alpha', 1e-4, 1000.0, log=True)}
    if name == 'ELASTIC_NET': return {'alpha': s('alpha', 1e-4, 100.0, log=True), 'l1_ratio': s('l1_ratio', 0.0, 1.0)}
    if name == 'LGBM':   return {'learning_rate': s('learning_rate', 0.01, 0.30), 'num_leaves': i('num_leaves', 8, 256),
                                 'min_child_samples': i('min_child_samples', 5, 200), 'reg_lambda': s('reg_lambda', 1e-3, 10.0, log=True),
                                 'subsample': s('subsample', 0.5, 1.0), 'colsample_bytree': s('colsample_bytree', 0.5, 1.0)}
    if name == 'XGBOOST':return {'learning_rate': s('learning_rate', 0.01, 0.30), 'max_depth': i('max_depth', 3, 10),
                                 'min_child_weight': i('min_child_weight', 5, 100), 'reg_lambda': s('reg_lambda', 1e-3, 10.0, log=True),
                                 'subsample': s('subsample', 0.5, 1.0), 'colsample_bytree': s('colsample_bytree', 0.5, 1.0)}
    if name == 'CATBOOST': return {'learning_rate': s('learning_rate', 0.01, 0.30), 'depth': i('depth', 3, 10),
                                   'l2_leaf_reg': s('l2_leaf_reg', 1e-3, 10.0, log=True)}
    if name == 'HGBR':   return {'learning_rate': s('learning_rate', 0.01, 0.30), 'max_depth': i('max_depth', 2, 15),
                                 'min_samples_leaf': i('min_samples_leaf', 5, 200), 'l2_regularization': s('l2_regularization', 1e-3, 10.0, log=True)}
    if name == 'GRADIENT_BOOST': return {'learning_rate': s('learning_rate', 0.01, 0.30), 'max_depth': i('max_depth', 2, 8),
                                         'subsample': s('subsample', 0.5, 1.0), 'min_samples_leaf': i('min_samples_leaf', 5, 100)}
    if name in ('RANDOM_FOREST', 'EXTRA_TREES'):
        return {'n_estimators': i('n_estimators', 50, 300), 'max_depth': i('max_depth', 3, 30),
                'max_features': s('max_features', 0.3, 1.0), 'min_samples_leaf': i('min_samples_leaf', 5, 100)}

def sbc_class(series_until_cut):
    """SBC (2005) on the product's history from its first sale: p = T / N (T = days since first sale, N = days with sales),
    CV2 of non-zero demand (sample variance, ddof=1); cut-offs 1.32 and 0.49. One sale only -> CV2 = 0. Fewer than 50 days since the first sale -> SHORT_HISTORY."""
    y = np.asarray(series_until_cut, float); nz = np.flatnonzero(y > 0)
    if len(nz) == 0: return 'NO_SALE', np.nan, np.nan
    y = y[nz[0]:]; pos = y[y > 0]
    p = len(y) / len(pos)
    if len(y) < 50: return 'SHORT_HISTORY', p, np.nan       # fewer than 50 days since the first sale
    cv2 = pos.var(ddof=1) / pos.mean() ** 2 if len(pos) > 1 else 0.0
    cls = ('SMOOTH' if cv2 <= 0.49 else 'ERRATIC') if p <= 1.32 else ('INTERMITTENT' if cv2 <= 0.49 else 'LUMPY')
    return cls, p, cv2
