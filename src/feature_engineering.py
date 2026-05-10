from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR, SRC_DIR, TARGET
from data_io import prepare_runtime
from validation import *


# =============================================================================
# 기본 피처 실험
# =============================================================================

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
fe_experiments_TARGET = 'avg_delay_minutes_next_30m'
fe_experiments_EPS = 1e-06
fe_experiments_LGBM_PARAMS = {'objective': 'mae', 'metric': 'mae', 'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 31, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
fe_experiments_RAW_ID_COLS = ['ID', 'scenario_id', 'layout_id']
fe_experiments_LAYOUT_FEATURES = ['pack_station_count', 'robot_total', 'charger_per_robot', 'robot_per_station', 'is_hub_spoke', 'is_narrow']
fe_experiments_ROLL_COLS = ['order_inflow_15m', 'battery_mean', 'congestion_score', 'pack_utilization', 'backorder_ratio']
fe_experiments_LAG_COLS = ['battery_mean', 'low_battery_ratio', 'order_inflow_15m', 'robot_utilization', 'robot_charging', 'congestion_score']
fe_experiments_DIFF_STEPS = [1, 2, 3, 5, 10]
fe_experiments_LAG_STEPS = list(range(1, 11))
fe_experiments_FULL_WINDOWS = list(range(2, 25))
fe_experiments_REP_WINDOWS = [3, 5, 7, 10, 15, 20, 24]
fe_experiments_REP_WINDOWS_TIGHT = [5, 10, 15, 20, 24]
fe_experiments_LONG_WINDOWS = [10, 15, 20, 24]

@dataclass
class fe_experiments_ExperimentConfig:
    name: str
    hypothesis: str
    include_lags: bool = True
    include_diffs: bool = True
    add_cumsum: bool = False
    add_decay: bool = False
    add_slope: bool = False
    add_persistence: bool = False
    remove_duplicate: bool = False
    remove_near_constant: bool = False
    remove_high_missing: bool = False
    remove_low_importance: bool = False
    roll_windows_keep: list[int] | None = None
    feature_mode: str = 'active'
    update_base: bool = True
    notes: dict = field(default_factory=dict)

def fe_experiments_safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b + fe_experiments_EPS)

def fe_experiments_add_layout_features(train: pd.DataFrame, test: pd.DataFrame, layout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    layout_use = layout[['layout_id', 'layout_type', 'pack_station_count', 'robot_total', 'charger_count']].copy()
    layout_use['charger_per_robot'] = fe_experiments_safe_div(layout_use['charger_count'], layout_use['robot_total'])
    layout_use['robot_per_station'] = fe_experiments_safe_div(layout_use['robot_total'], layout_use['pack_station_count'])
    layout_use['is_hub_spoke'] = (layout_use['layout_type'] == 'hub_spoke').astype(int)
    layout_use['is_narrow'] = (layout_use['layout_type'] == 'narrow').astype(int)
    layout_use = layout_use.drop(columns=['layout_type', 'charger_count'])
    return (train.merge(layout_use, on='layout_id', how='left'), test.merge(layout_use, on='layout_id', how='left'))

def fe_experiments_add_base_bottleneck_features(df: pd.DataFrame) -> None:
    available_robot = df['robot_total'] - df['robot_charging']
    available_robot_ratio = fe_experiments_safe_div(available_robot, df['robot_total'])
    order_per_pack_station = fe_experiments_safe_div(df['order_inflow_15m'], df['pack_station_count'])
    new_cols = {'pack_saturation': (df['pack_utilization'] >= 0.95).astype(int), 'pack_gap_to_full': 1 - df['pack_utilization'], 'order_per_pack_station': order_per_pack_station, 'available_robot': available_robot, 'available_robot_ratio': available_robot_ratio, 'order_per_available_robot': fe_experiments_safe_div(df['order_inflow_15m'], available_robot), 'idle_x_pack_utilization': df['robot_idle'] * df['pack_utilization'], 'charging_ratio': fe_experiments_safe_div(df['robot_charging'], df['robot_total']), 'item_inflow_15m': df['order_inflow_15m'] * df['avg_items_per_order'], 'sku_per_order': fe_experiments_safe_div(df['unique_sku_15m'], df['order_inflow_15m']), 'sku_pressure': df['unique_sku_15m'] * df['order_inflow_15m'], 'heavy_item_pressure': df['heavy_item_ratio'] * df['order_inflow_15m'], 'robot_shortage_ratio': 1 - available_robot_ratio, 'items_per_pack_station': fe_experiments_safe_div(df['order_inflow_15m'] * df['avg_items_per_order'], df['pack_station_count']), 'heavy_order_per_pack_station': fe_experiments_safe_div(df['order_inflow_15m'] * df['heavy_item_ratio'], df['pack_station_count']), 'sku_per_pack_station': fe_experiments_safe_div(df['unique_sku_15m'], df['pack_station_count']), 'order_per_pack_x_pack_utilization': order_per_pack_station * df['pack_utilization'], 'item_per_available_robot': fe_experiments_safe_div(df['order_inflow_15m'] * df['avg_items_per_order'], available_robot), 'trip_pressure_per_available_robot': fe_experiments_safe_div(df['avg_trip_distance'] * df['order_inflow_15m'], available_robot), 'congestion_x_avg_trip_distance': df['congestion_score'] * df['avg_trip_distance']}
    for col, values in new_cols.items():
        df[col] = values

def fe_experiments_add_lag_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col in fe_experiments_LAG_COLS:
        group = df.groupby('scenario_id', sort=False)[col]
        lag_cache = {}
        for lag in fe_experiments_LAG_STEPS:
            lag_cache[lag] = group.shift(lag)
            new_cols[f'{col}_lag{lag}'] = lag_cache[lag]
        for lag in fe_experiments_DIFF_STEPS:
            new_cols[f'{col}_diff{lag}'] = df[col] - lag_cache[lag]
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def fe_experiments_rolling_series(shifted: pd.Series, groups: pd.Series, window: int, how: str) -> pd.Series:
    roller = shifted.groupby(groups, sort=False).rolling(window, min_periods=1)
    if how == 'mean':
        out = roller.mean()
    elif how == 'max':
        out = roller.max()
    elif how == 'std':
        out = shifted.groupby(groups, sort=False).rolling(window, min_periods=2).std()
    else:
        raise ValueError(how)
    return out.reset_index(level=0, drop=True).sort_index()

def fe_experiments_add_rolling_features(df: pd.DataFrame, windows: Iterable[int]=fe_experiments_FULL_WINDOWS) -> pd.DataFrame:
    new_cols = {}
    groups = df['scenario_id']
    for col in fe_experiments_ROLL_COLS:
        shifted = df.groupby('scenario_id', sort=False)[col].shift(1)
        for window in windows:
            new_cols[f'{col}_roll{window}_mean'] = fe_experiments_rolling_series(shifted, groups, window, 'mean')
            new_cols[f'{col}_roll{window}_max'] = fe_experiments_rolling_series(shifted, groups, window, 'max')
            new_cols[f'{col}_roll{window}_std'] = fe_experiments_rolling_series(shifted, groups, window, 'std')
    shifted_opp = df.groupby('scenario_id', sort=False)['order_per_pack_station'].shift(1)
    for window in windows:
        cong_mean = f'congestion_score_roll{window}_mean'
        pack_mean = f'pack_utilization_roll{window}_mean'
        order_mean = f'order_inflow_15m_roll{window}_mean'
        opp_mean = fe_experiments_rolling_series(shifted_opp, groups, window, 'mean')
        opp_max = fe_experiments_rolling_series(shifted_opp, groups, window, 'max')
        new_cols[f'congestion_vs_roll{window}'] = df['congestion_score'] - new_cols[cong_mean]
        new_cols[f'congestion_roll{window}_x_pack'] = new_cols[cong_mean] * df['pack_utilization']
        new_cols[f'robot_shortage_x_congestion_roll{window}'] = df['robot_shortage_ratio'] * new_cols[cong_mean]
        new_cols[f'pack_utilization_vs_roll{window}'] = df['pack_utilization'] - new_cols[pack_mean]
        new_cols[f'order_per_pack_station_roll{window}_mean'] = opp_mean
        new_cols[f'order_per_pack_station_roll{window}_max'] = opp_max
        new_cols[f'order_per_pack_station_vs_roll{window}'] = df['order_per_pack_station'] - opp_mean
        new_cols[f'order_roll{window}_per_pack_station'] = fe_experiments_safe_div(new_cols[order_mean], df['pack_station_count'])
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def fe_experiments_add_cumsum_features(df: pd.DataFrame) -> list[str]:
    created = []
    mapping = {'order_cumsum': 'order_inflow_15m', 'congestion_cumsum': 'congestion_score', 'pack_utilization_cumsum': 'pack_utilization', 'backorder_ratio_cumsum': 'backorder_ratio'}
    for out_col, src_col in mapping.items():
        shifted = df.groupby('scenario_id', sort=False)[src_col].shift(1).fillna(0)
        df[out_col] = shifted.groupby(df['scenario_id'], sort=False).cumsum()
        created.append(out_col)
    return created

def fe_experiments_add_decay_features(df: pd.DataFrame) -> list[str]:
    created = []
    cols = ['order_inflow_15m', 'congestion_score', 'pack_utilization', 'backorder_ratio', 'order_per_pack_station']
    for col in cols:
        shifted = df.groupby('scenario_id', sort=False)[col].shift(1).fillna(0)
        for decay in [0.7, 0.85, 0.95]:
            suffix = str(decay).replace('.', '')
            out_col = f'{col}_decay{suffix}'
            df[out_col] = shifted.groupby(df['scenario_id'], sort=False).transform(lambda s, d=decay: s.ewm(alpha=1 - d, adjust=False).mean())
            created.append(out_col)
    return created

def fe_experiments_add_slope_features(df: pd.DataFrame) -> list[str]:
    created = []
    x = np.arange(5, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())

    def slope_last5(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        if len(y) < 2:
            return 0.0
        x_part = x_centered[-len(y):]
        return float(np.sum((y - y.mean()) * x_part) / np.sum(x_part ** 2)) if np.sum(x_part ** 2) else 0.0
    for col in ['congestion_score', 'pack_utilization', 'order_inflow_15m', 'backorder_ratio']:
        shifted = df.groupby('scenario_id', sort=False)[col].shift(1)
        out_col = f'{col}_slope5'
        df[out_col] = shifted.groupby(df['scenario_id'], sort=False).transform(lambda s: s.rolling(5, min_periods=2).apply(slope_last5, raw=True))
        created.append(out_col)
    return created

def fe_experiments_add_persistence_features(df: pd.DataFrame) -> list[str]:
    created = []
    conditions = {'congestion_high': df['congestion_score'] > 0.8, 'pack_high': df['pack_utilization'] > 0.9, 'backorder_high': df['backorder_ratio'] > 0, 'robot_shortage_high': df['robot_shortage_ratio'] > 0.2}
    for name, condition in conditions.items():
        state = condition.astype(int)
        shifted = state.groupby(df['scenario_id'], sort=False).shift(1).fillna(0)
        run_col = f'{name}_run_length'
        pieces = []
        for _, s in shifted.groupby(df['scenario_id'], sort=False):
            breaks = (s == 0).cumsum()
            pieces.append(s.groupby(breaks).cumsum())
        df[run_col] = pd.concat(pieces).sort_index()
        created.append(run_col)
        for window in [5, 10, 15, 24]:
            count_col = f'{name}_count_roll{window}'
            cross_col = f'{name}_cross_roll{window}'
            grouped = shifted.groupby(df['scenario_id'], sort=False)
            df[count_col] = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            prev = shifted.groupby(df['scenario_id'], sort=False).shift(1).fillna(0)
            crossing = ((shifted == 1) & (prev == 0)).astype(int)
            df[cross_col] = crossing.groupby(df['scenario_id'], sort=False).transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            created.extend([count_col, cross_col])
    return created

def fe_experiments_build_feature_store() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    raw_feature_cols = [c for c in train.columns if c not in fe_experiments_RAW_ID_COLS + [fe_experiments_TARGET]]
    train, test = fe_experiments_add_layout_features(train, test, layout)
    for df in [train, test]:
        fe_experiments_add_base_bottleneck_features(df)
        df.sort_values(['scenario_id', 'ID'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['timeslot'] = df.groupby('scenario_id', sort=False).cumcount()
        df = fe_experiments_add_lag_diff_features(df)
        df = fe_experiments_add_rolling_features(df)
        if fe_experiments_TARGET in df.columns:
            train = df
        else:
            test = df
    metadata = {'raw_feature_cols': raw_feature_cols, 'layout_features': fe_experiments_LAYOUT_FEATURES, 'baseline_lag_steps': fe_experiments_LAG_STEPS, 'baseline_diff_steps': fe_experiments_DIFF_STEPS, 'roll_windows': fe_experiments_FULL_WINDOWS}
    return (train, test, metadata)

def fe_experiments_ensure_optional_features(train: pd.DataFrame, test: pd.DataFrame, config: ExperimentConfig, optional_groups: dict) -> None:
    for name, enabled, fn in [('cumsum', config.add_cumsum, fe_experiments_add_cumsum_features), ('decay', config.add_decay, fe_experiments_add_decay_features), ('slope', config.add_slope, fe_experiments_add_slope_features), ('persistence', config.add_persistence, fe_experiments_add_persistence_features)]:
        if enabled and name not in optional_groups:
            train_cols = fn(train)
            test_cols = fn(test)
            assert train_cols == test_cols
            optional_groups[name] = train_cols

def fe_experiments_is_lag_col(col: str) -> bool:
    return any((col.startswith(f'{base}_lag') for base in fe_experiments_LAG_COLS))

def fe_experiments_is_diff_col(col: str) -> bool:
    return any((col.startswith(f'{base}_diff') for base in fe_experiments_LAG_COLS))

def fe_experiments_is_roll_col(col: str) -> bool:
    return '_roll' in col or '_vs_roll' in col

def fe_experiments_extract_roll_window(col: str) -> int | None:
    marker = 'roll'
    if marker not in col:
        return None
    tail = col.split(marker, 1)[1]
    digits = ''
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None

def fe_experiments_select_columns(train: pd.DataFrame, config: ExperimentConfig, active_columns: list[str], optional_groups: dict, prior_low_importance: set[str], raw_feature_cols: list[str]) -> tuple[list[str], dict]:
    info = {'removed': []}
    if config.feature_mode == 'roll_only':
        cols = [c for c in train.columns if fe_experiments_is_roll_col(c)]
    elif config.feature_mode == 'roll_base':
        cols = [c for c in train.columns if c in raw_feature_cols or fe_experiments_is_roll_col(c)]
    elif config.feature_mode == 'roll_base_layout':
        cols = [c for c in train.columns if c in raw_feature_cols or c in fe_experiments_LAYOUT_FEATURES or fe_experiments_is_roll_col(c)]
    else:
        cols = list(active_columns)
        for group in ['cumsum', 'decay', 'slope', 'persistence']:
            if getattr(config, f'add_{group}'):
                cols.extend(optional_groups[group])
    cols = [c for c in cols if c not in fe_experiments_RAW_ID_COLS + [fe_experiments_TARGET]]
    if not config.include_lags:
        removed = [c for c in cols if fe_experiments_is_lag_col(c)]
        cols = [c for c in cols if not fe_experiments_is_lag_col(c)]
        info['removed'].extend(removed)
    if not config.include_diffs:
        removed = [c for c in cols if fe_experiments_is_diff_col(c)]
        cols = [c for c in cols if not fe_experiments_is_diff_col(c)]
        info['removed'].extend(removed)
    if config.roll_windows_keep is not None:
        keep = set(config.roll_windows_keep)
        removed = [c for c in cols if fe_experiments_is_roll_col(c) and fe_experiments_extract_roll_window(c) not in keep]
        cols = [c for c in cols if not (fe_experiments_is_roll_col(c) and fe_experiments_extract_roll_window(c) not in keep)]
        info['removed'].extend(removed)
    if config.remove_duplicate:
        dupes = fe_experiments_find_duplicate_columns(train[cols])
        cols = [c for c in cols if c not in dupes]
        info['removed'].extend(sorted(dupes))
    if config.remove_near_constant:
        low_var = fe_experiments_find_near_constant_columns(train[cols])
        cols = [c for c in cols if c not in low_var]
        info['removed'].extend(sorted(low_var))
    if config.remove_high_missing:
        high_missing = fe_experiments_find_high_missing_columns(train[cols])
        cols = [c for c in cols if c not in high_missing]
        info['removed'].extend(sorted(high_missing))
    if config.remove_low_importance:
        protected = {c for c in cols if fe_experiments_is_roll_col(c) or any((c in optional_groups.get(g, []) for g in ['cumsum', 'decay', 'persistence']))}
        removable = prior_low_importance - protected
        cols = [c for c in cols if c not in removable]
        info['removed'].extend(sorted(removable))
    return (list(dict.fromkeys(cols)), info)

def fe_experiments_find_duplicate_columns(x: pd.DataFrame) -> set[str]:
    seen = {}
    dupes = set()
    for col in x.columns:
        signature = tuple(pd.util.hash_pandas_object(x[col], index=False).values)
        if signature in seen:
            dupes.add(col)
        else:
            seen[signature] = col
    return dupes

def fe_experiments_find_near_constant_columns(x: pd.DataFrame) -> set[str]:
    out = set()
    n = len(x)
    for col in x.columns:
        vc = x[col].value_counts(dropna=False)
        if len(vc) <= 1 or (len(vc) and vc.iloc[0] / n >= 0.999):
            out.add(col)
    return out

def fe_experiments_find_high_missing_columns(x: pd.DataFrame) -> set[str]:
    return set(x.columns[x.isna().mean() >= 0.95])

def fe_experiments_fit_predict_lgbm(model: lgb.LGBMRegressor, x_tr, y_tr, x_va, y_va):
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(model.predict(x_va), 0, None)
    return (model, pred)

def fe_experiments_run_experiment(config: ExperimentConfig, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], output_root: Path, metadata: dict) -> dict:
    output_dir = output_root / config.name
    report_dir = output_dir / 'validation_reports'
    submission_dir = output_dir / 'submissions'
    oof_dir = output_dir / 'oof_predictions'
    importance_dir = output_dir / 'feature_importance'
    for path in [report_dir, submission_dir, oof_dir, importance_dir]:
        path.mkdir(parents=True, exist_ok=True)
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
    x_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
    y = train[fe_experiments_TARGET].astype(float)
    groups = train['scenario_id'].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(x), dtype=float)
    test_pred = np.zeros(len(x_test), dtype=float)
    fold_rows = []
    importances = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
        model, pred = fe_experiments_fit_predict_lgbm(model, x.iloc[tr_idx], y.iloc[tr_idx], x.iloc[va_idx], y.iloc[va_idx])
        fold_test = np.clip(model.predict(x_test), 0, None)
        oof[va_idx] = pred
        test_pred += fold_test / 5
        fold_rows.append({'model_name': config.name, 'validation': 'groupkfold', 'fold': fold, 'n_train': len(tr_idx), 'n_valid': len(va_idx), 'valid_target_mean': float(y.iloc[va_idx].mean()), 'valid_target_max': float(y.iloc[va_idx].max()), 'pred_mean': float(pred.mean()), 'pred_max': float(pred.max()), 'mae': float(mean_absolute_error(y.iloc[va_idx], pred)), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators'])})
        importances.append(pd.DataFrame({'model_name': config.name, 'feature': feature_cols, 'gain': model.booster_.feature_importance(importance_type='gain'), 'split': model.booster_.feature_importance(importance_type='split'), 'fold': fold}))
    group_oof = val_linear_make_prediction_frame(y.values, oof, groups=groups)
    group_summary = val_linear_summarize_prediction_frame(group_oof, 'groupkfold', config.name)
    group_bin = val_linear_make_bin_report(group_oof, 'groupkfold', config.name)
    fold_report = pd.DataFrame(fold_rows)
    target_heavy_holdout_config = val_linear_TargetHeavyConfig(random_state=42)
    train_mask, valid_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, target_heavy_holdout_config)
    exp_model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
    exp_model, exp_pred = fe_experiments_fit_predict_lgbm(exp_model, x.loc[train_mask], y.loc[train_mask], x.loc[valid_mask], y.loc[valid_mask])
    exp_oof = val_linear_make_prediction_frame(y.loc[valid_mask].values, exp_pred, groups=groups[valid_mask])
    exp_summary = val_linear_summarize_prediction_frame(exp_oof, 'target_heavy_target_heavy_holdout', config.name)
    for key, value in asdict(target_heavy_holdout_config).items():
        exp_summary[key] = value
    exp_fold = pd.DataFrame([{'model_name': config.name, 'validation': 'target_heavy_target_heavy_holdout', 'n_train': int(train_mask.sum()), 'n_valid': int(valid_mask.sum()), 'n_train_groups': len(np.unique(groups[train_mask])), 'n_valid_groups': len(np.unique(groups[valid_mask])), 'valid_target_mean': float(y.loc[valid_mask].mean()), 'valid_target_max': float(y.loc[valid_mask].max()), 'pred_mean': float(exp_pred.mean()), 'pred_max': float(exp_pred.max()), 'mae': float(mean_absolute_error(y.loc[valid_mask], exp_pred)), 'best_iteration': int(exp_model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators'])}])
    exp_bin = val_linear_make_bin_report(exp_oof, 'target_heavy_target_heavy_holdout', config.name)
    summary = pd.concat([group_summary, exp_summary], ignore_index=True)
    fold_report = pd.concat([fold_report, exp_fold], ignore_index=True, sort=False)
    bin_report = pd.concat([group_bin, exp_bin], ignore_index=True)
    summary.to_csv(report_dir / f'{config.name}_summary.csv', index=False)
    fold_report.to_csv(report_dir / f'{config.name}_fold_report.csv', index=False)
    bin_report.to_csv(report_dir / f'{config.name}_bin_report.csv', index=False)
    group_oof.to_csv(oof_dir / f'{config.name}_groupkfold_oof.csv', index=False)
    exp_oof.to_csv(oof_dir / f'{config.name}_target_heavy_target_heavy_holdout_oof.csv', index=False)
    scenario_stat.to_csv(report_dir / f'{config.name}_scenario_target_stat.csv', index=False)
    importance = pd.concat(importances, ignore_index=True)
    importance.to_csv(importance_dir / f'{config.name}_feature_importance_by_fold.csv', index=False)
    importance_mean = importance.groupby('feature', as_index=False)[['gain', 'split']].mean().sort_values('gain', ascending=False)
    importance_mean.to_csv(importance_dir / f'{config.name}_feature_importance_mean.csv', index=False)
    submission = pd.read_csv('sample_submission.csv')[['ID']]
    pred_frame = pd.DataFrame({'ID': test['ID'].values, fe_experiments_TARGET: np.clip(test_pred, 0, None)})
    submission = submission.merge(pred_frame, on='ID', how='left')
    submission[fe_experiments_TARGET] = submission[fe_experiments_TARGET].clip(lower=0)
    submission.to_csv(submission_dir / f'{config.name}_submission.csv', index=False)
    config_payload = {'experiment': asdict(config), 'lgbm_params': fe_experiments_LGBM_PARAMS, 'feature_count': len(feature_cols), 'features': feature_cols, 'metadata': metadata, 'target_bins': val_linear_TARGET_BINS, 'target_bin_labels': val_linear_TARGET_BIN_LABELS}
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)
    group_mae = float(summary.loc[summary['validation'] == 'groupkfold', 'mae'].iloc[0])
    target_heavy_holdout_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'mae'].iloc[0])
    high50_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high50_mae'].iloc[0])
    high100_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high100_mae'].iloc[0])
    return {'name': config.name, 'feature_count': len(feature_cols), 'groupkfold_mae': group_mae, 'target_heavy_holdout_mae': target_heavy_holdout_mae, 'high50_mae': high50_mae, 'high100_mae': high100_mae, 'importance_mean': importance_mean}

def fe_experiments_append_history(output_root: Path, row: dict) -> None:
    path = output_root / 'experiment_history.csv'
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)

def fe_experiments_build_experiment_plan() -> list[fe_experiments_ExperimentConfig]:
    return [fe_experiments_ExperimentConfig('baseline_lag_diff_roll', '현재까지 알려진 local baseline입니다.', update_base=False), fe_experiments_ExperimentConfig('diff_only_no_lag', 'lag 입력을 제거하고 diff와 rolling 피처만 유지합니다.', include_lags=False), fe_experiments_ExperimentConfig('cumulative_pressure', '시나리오 전체 과거 이력 기반 누적 pressure 피처를 추가합니다.', add_cumsum=True), fe_experiments_ExperimentConfig('decayed_pressure', '지수 감쇠 방식의 병목 pressure 피처를 추가합니다.', add_decay=True), fe_experiments_ExperimentConfig('slope_trend', '최근 병목 관련 축의 추세 기울기 피처를 추가합니다.', add_slope=True), fe_experiments_ExperimentConfig('state_persistence', '높은 상태가 지속되는 시간과 threshold persistence 피처를 추가합니다.', add_persistence=True), fe_experiments_ExperimentConfig('exact_duplicate_removal', '완전히 중복된 컬럼을 제거합니다.', remove_duplicate=True), fe_experiments_ExperimentConfig('near_constant_removal', '거의 변하지 않는 컬럼을 제거합니다.', remove_near_constant=True), fe_experiments_ExperimentConfig('high_missing_removal', '결측률이 매우 높은 컬럼을 제거합니다.', remove_high_missing=True), fe_experiments_ExperimentConfig('repeated_low_importance_removal', '반복적으로 중요도가 낮거나 0인 비보호 컬럼을 제거합니다.', remove_low_importance=True), fe_experiments_ExperimentConfig('representative_roll_windows', '대표 rolling window만 유지합니다.', roll_windows_keep=fe_experiments_REP_WINDOWS), fe_experiments_ExperimentConfig('tight_roll_windows', '더 좁게 고른 대표 rolling window만 유지합니다.', roll_windows_keep=fe_experiments_REP_WINDOWS_TIGHT), fe_experiments_ExperimentConfig('long_roll_windows', '긴 rolling window만 유지합니다.', roll_windows_keep=fe_experiments_LONG_WINDOWS), fe_experiments_ExperimentConfig('roll_only', 'rolling으로 생성된 피처만 사용해 학습합니다.', feature_mode='roll_only', update_base=False), fe_experiments_ExperimentConfig('roll_plus_raw_base', '원본 train 피처와 rolling 생성 피처를 함께 사용해 학습합니다.', feature_mode='roll_base', update_base=False), fe_experiments_ExperimentConfig('target_heavy_holdout_roll_base_layout', '원본 train, rolling, 선택된 layout 피처를 함께 사용해 학습합니다.', feature_mode='roll_base_layout', update_base=False)]

def fe_experiments_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', default='outputs')
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    train, test, metadata = fe_experiments_build_feature_store()
    raw_feature_cols = metadata['raw_feature_cols']
    base_columns = [c for c in train.columns if c not in fe_experiments_RAW_ID_COLS + [fe_experiments_TARGET]]
    active_columns = list(base_columns)
    optional_groups: dict[str, list[str]] = {}
    prior_low_importance: set[str] = set()
    best_target_heavy_holdout = np.inf
    best_name = None
    for config in fe_experiments_build_experiment_plan():
        if args.only and config.name not in args.only:
            continue
        print(f'\n===== Running {config.name} =====', flush=True)
        fe_experiments_ensure_optional_features(train, test, config, optional_groups)
        feature_cols, selection_info = fe_experiments_select_columns(train=train, config=config, active_columns=active_columns, optional_groups=optional_groups, prior_low_importance=prior_low_importance, raw_feature_cols=raw_feature_cols)
        config.notes['selection'] = {'removed_count': len(selection_info['removed']), 'removed_sample': selection_info['removed'][:50]}
        result = fe_experiments_run_experiment(config, train, test, feature_cols, output_root, metadata)
        improved = result['target_heavy_holdout_mae'] < best_target_heavy_holdout
        if improved:
            best_target_heavy_holdout = result['target_heavy_holdout_mae']
            best_name = config.name
            if config.update_base:
                active_columns = feature_cols
        imp = result['importance_mean']
        zero_gain = set(imp.loc[imp['gain'] <= 0, 'feature'])
        low_tail = set(imp.tail(max(20, int(len(imp) * 0.05)))['feature'])
        prior_low_importance |= zero_gain & low_tail
        row = {'experiment_name': config.name, 'base_after_run': best_name, 'hypothesis': config.hypothesis, 'feature_count': result['feature_count'], 'groupkfold_mae': result['groupkfold_mae'], 'target_heavy_holdout_mae': result['target_heavy_holdout_mae'], 'high50_mae': result['high50_mae'], 'high100_mae': result['high100_mae'], 'improved_target_heavy_holdout': improved, 'best_target_heavy_holdout_after_run': best_target_heavy_holdout, 'removed_count': len(selection_info['removed'])}
        fe_experiments_append_history(output_root, row)
        print(f"{config.name}: target_heavy_holdout={result['target_heavy_holdout_mae']:.6f}, group={result['groupkfold_mae']:.6f}, features={result['feature_count']}, improved={improved}", flush=True)
    print(f'\nDone. best_target_heavy_holdout={best_target_heavy_holdout:.6f} ({best_name})', flush=True)

# =============================================================================
# neighbor 피처와 결측 처리 실험
# =============================================================================

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT = 10.764679
fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS = [2, 13, 20, 21]
fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS = {1, 2, 3, 5, 10}
fe_neighbor_feature_missing_exps_TARGET = fe_experiments_TARGET
fe_neighbor_feature_missing_exps_OUTPUT_ROOT = Path('outputs')

def fe_neighbor_feature_missing_exps_filter_lag_diff(cols: list[str]) -> list[str]:
    out = []
    for col in cols:
        if fe_experiments_is_lag_col(col):
            if int(col.rsplit('lag', 1)[1]) in fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS:
                out.append(col)
            continue
        if fe_experiments_is_diff_col(col):
            if int(col.rsplit('diff', 1)[1]) in fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS:
                out.append(col)
            continue
        out.append(col)
    return out

def fe_neighbor_feature_missing_exps_best_columns(train: pd.DataFrame, metadata: dict) -> list[str]:
    cfg = fe_experiments_ExperimentConfig(name='neighbor_feature_base', hypothesis='best feature base', roll_windows_keep=fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, update_base=False)
    cols, _ = fe_experiments_select_columns(train=train, config=cfg, active_columns=[c for c in train.columns if c not in fe_experiments_RAW_ID_COLS + [fe_neighbor_feature_missing_exps_TARGET]], optional_groups={}, prior_low_importance=set(), raw_feature_cols=metadata['raw_feature_cols'])
    return fe_neighbor_feature_missing_exps_filter_lag_diff(cols)

def fe_neighbor_feature_missing_exps_q(train: pd.DataFrame, col: str, quantile: float) -> float:
    return float(train[col].replace([np.inf, -np.inf], np.nan).quantile(quantile))

def fe_neighbor_feature_missing_exps_add_neighbor_core_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    p13 = 'pack_utilization_roll13_mean'
    p20 = 'pack_utilization_roll20_mean'
    p21 = 'pack_utilization_roll21_mean'
    p13_std = 'pack_utilization_roll13_std'
    b21_mean = 'battery_mean_roll21_mean'
    b21_std = 'battery_mean_roll21_std'
    rc13 = 'robot_shortage_x_congestion_roll13'
    rc20 = 'robot_shortage_x_congestion_roll20'
    rc21 = 'robot_shortage_x_congestion_roll21'
    thresholds = {'pack_q80': fe_neighbor_feature_missing_exps_q(train, p21, 0.8), 'pack13_std_q80': fe_neighbor_feature_missing_exps_q(train, p13_std, 0.8), 'battery_mean_q30': fe_neighbor_feature_missing_exps_q(train, b21_mean, 0.3), 'robot_cong13_q80': fe_neighbor_feature_missing_exps_q(train, rc13, 0.8), 'robot_cong20_q80': fe_neighbor_feature_missing_exps_q(train, rc20, 0.8), 'robot_cong21_q80': fe_neighbor_feature_missing_exps_q(train, rc21, 0.8)}
    created = []
    for df in [train, test]:
        pack_stack = df[[p13, p20, p21]]
        robot_stack = df[[rc13, rc20, rc21]]
        df['nf_pack_long_mean_max'] = pack_stack.max(axis=1)
        df['nf_pack_long_mean_gap'] = pack_stack.max(axis=1) - pack_stack.min(axis=1)
        df['nf_pack_high_mean_and_volatility'] = ((df[p13] >= thresholds['pack_q80']) & (df[p13_std] >= thresholds['pack13_std_q80'])).astype(int)
        df['nf_battery_roll21_instability'] = df[b21_std] / (df[b21_mean].abs() + fe_experiments_EPS)
        df['nf_battery_low_and_pack_high'] = ((df[b21_mean] <= thresholds['battery_mean_q30']) & (df[p21] >= thresholds['pack_q80'])).astype(int)
        df['nf_robot_cong_long_max'] = robot_stack.max(axis=1)
        df['nf_robot_cong_long_mean'] = robot_stack.mean(axis=1)
        df['nf_robot_cong_persistent_high'] = ((df[rc13] >= thresholds['robot_cong13_q80']) & (df[rc20] >= thresholds['robot_cong20_q80']) & (df[rc21] >= thresholds['robot_cong21_q80'])).astype(int)
        df['nf_pack_robot_cong_pressure'] = df[p21] * df[rc21]
        df['nf_pack_persistent_pressure'] = pack_stack.mean(axis=1) / (pack_stack.std(axis=1) + fe_experiments_EPS)
    created = ['nf_pack_long_mean_max', 'nf_pack_long_mean_gap', 'nf_pack_high_mean_and_volatility', 'nf_battery_roll21_instability', 'nf_battery_low_and_pack_high', 'nf_robot_cong_long_max', 'nf_robot_cong_long_mean', 'nf_robot_cong_persistent_high', 'nf_pack_robot_cong_pressure', 'nf_pack_persistent_pressure']
    return created

def fe_neighbor_feature_missing_exps_add_neighbor_extra_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    pcols = ['pack_utilization_roll13_mean', 'pack_utilization_roll20_mean', 'pack_utilization_roll21_mean']
    rcols = ['robot_shortage_x_congestion_roll13', 'robot_shortage_x_congestion_roll20', 'robot_shortage_x_congestion_roll21']
    bmean = 'battery_mean_roll21_mean'
    bstd = 'battery_mean_roll21_std'
    created = []
    pack_q85 = fe_neighbor_feature_missing_exps_q(train, 'pack_utilization_roll21_mean', 0.85)
    robot_q85 = fe_neighbor_feature_missing_exps_q(train, 'robot_shortage_x_congestion_roll21', 0.85)
    battery_instab_q85 = fe_neighbor_feature_missing_exps_q(train.assign(_tmp=train[bstd] / (train[bmean].abs() + fe_experiments_EPS)), '_tmp', 0.85)
    for df in [train, test]:
        pack = df[pcols]
        robot = df[rcols]
        df['nf_extra_pack_long_mean_avg'] = pack.mean(axis=1)
        df['nf_extra_pack_long_mean_std'] = pack.std(axis=1)
        df['nf_extra_pack_high_count'] = (pack >= pack_q85).sum(axis=1)
        df['nf_extra_robot_cong_long_std'] = robot.std(axis=1)
        df['nf_extra_pack_x_robot_cong_mean'] = pack.mean(axis=1) * robot.mean(axis=1)
        df['nf_extra_pack_high_robot_high'] = ((pack.mean(axis=1) >= pack_q85) & (robot.mean(axis=1) >= robot_q85)).astype(int)
        instab = df[bstd] / (df[bmean].abs() + fe_experiments_EPS)
        df['nf_extra_battery_instab_high'] = (instab >= battery_instab_q85).astype(int)
        df['nf_extra_pack_robot_battery_risk'] = ((pack.mean(axis=1) >= pack_q85) & (robot.mean(axis=1) >= robot_q85) & (instab >= battery_instab_q85)).astype(int)
    created = ['nf_extra_pack_long_mean_avg', 'nf_extra_pack_long_mean_std', 'nf_extra_pack_high_count', 'nf_extra_robot_cong_long_std', 'nf_extra_pack_x_robot_cong_mean', 'nf_extra_pack_high_robot_high', 'nf_extra_battery_instab_high', 'nf_extra_pack_robot_battery_risk']
    return created

def fe_neighbor_feature_missing_exps_fill_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    xt = test[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    if strategy == 'default_minus999':
        return (x.fillna(-999), xt.fillna(-999))
    if strategy in {'lag_roll_ffill_bfill', 'lag_roll_linear_interpolate', 'roll_std_zero'}:
        lag_roll_cols = [c for c in feature_cols if '_lag' in c or '_diff' in c or '_roll' in c]
        roll_std_cols = [c for c in feature_cols if '_roll' in c and '_std' in c]
        if strategy == 'lag_roll_ffill_bfill':
            for frame, source in [(x, train), (xt, test)]:
                cols = [c for c in lag_roll_cols if c in frame.columns]
                frame[cols] = frame[cols].groupby(source['scenario_id'], sort=False).transform(lambda s: s.ffill().bfill())
        elif strategy == 'lag_roll_linear_interpolate':
            for frame, source in [(x, train), (xt, test)]:
                cols = [c for c in lag_roll_cols if c in frame.columns]
                frame[cols] = frame[cols].groupby(source['scenario_id'], sort=False).transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
        elif strategy == 'roll_std_zero':
            for frame in [x, xt]:
                cols = [c for c in roll_std_cols if c in frame.columns]
                frame[cols] = frame[cols].fillna(0)
        return (x.fillna(-999), xt.fillna(-999))
    raise ValueError(strategy)

def fe_neighbor_feature_missing_exps_run_experiment_with_fill(name: str, hypothesis: str, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], metadata: dict, fill_strategy: str) -> dict:
    output_dir = fe_neighbor_feature_missing_exps_OUTPUT_ROOT / name
    report_dir = output_dir / 'validation_reports'
    submission_dir = output_dir / 'submissions'
    oof_dir = output_dir / 'oof_predictions'
    importance_dir = output_dir / 'feature_importance'
    for path in [report_dir, submission_dir, oof_dir, importance_dir]:
        path.mkdir(parents=True, exist_ok=True)
    x, xt = fe_neighbor_feature_missing_exps_fill_features(train, test, feature_cols, fill_strategy)
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(x))
    test_pred = np.zeros(len(xt))
    fold_rows = []
    importances = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
        model.fit(x.iloc[tr_idx], y.iloc[tr_idx], eval_set=[(x.iloc[va_idx], y.iloc[va_idx])], callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = np.clip(model.predict(x.iloc[va_idx]), 0, None)
        oof[va_idx] = pred
        test_pred += np.clip(model.predict(xt), 0, None) / 5
        fold_rows.append({'model_name': name, 'validation': 'groupkfold', 'fold': fold, 'mae': float(mean_absolute_error(y.iloc[va_idx], pred)), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators'])})
        importances.append(pd.DataFrame({'model_name': name, 'feature': feature_cols, 'gain': model.booster_.feature_importance(importance_type='gain'), 'split': model.booster_.feature_importance(importance_type='split'), 'fold': fold}))
    group_oof = val_linear_make_prediction_frame(y.values, oof, groups=groups)
    group_summary = val_linear_summarize_prediction_frame(group_oof, 'groupkfold', name)
    group_bin = val_linear_make_bin_report(group_oof, 'groupkfold', name)
    exp_cfg = val_linear_TargetHeavyConfig(random_state=42)
    tr_mask, va_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, exp_cfg)
    exp_model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
    exp_model.fit(x.loc[tr_mask], y.loc[tr_mask], eval_set=[(x.loc[va_mask], y.loc[va_mask])], callbacks=[lgb.early_stopping(50, verbose=False)])
    exp_pred = np.clip(exp_model.predict(x.loc[va_mask]), 0, None)
    exp_oof = val_linear_make_prediction_frame(y.loc[va_mask].values, exp_pred, groups=groups[va_mask])
    exp_summary = val_linear_summarize_prediction_frame(exp_oof, 'target_heavy_target_heavy_holdout', name)
    exp_bin = val_linear_make_bin_report(exp_oof, 'target_heavy_target_heavy_holdout', name)
    summary = pd.concat([group_summary, exp_summary], ignore_index=True)
    bin_report = pd.concat([group_bin, exp_bin], ignore_index=True)
    fold_report = pd.DataFrame(fold_rows)
    summary.to_csv(report_dir / f'{name}_summary.csv', index=False)
    fold_report.to_csv(report_dir / f'{name}_fold_report.csv', index=False)
    bin_report.to_csv(report_dir / f'{name}_bin_report.csv', index=False)
    group_oof.to_csv(oof_dir / f'{name}_groupkfold_oof.csv', index=False)
    exp_oof.to_csv(oof_dir / f'{name}_target_heavy_target_heavy_holdout_oof.csv', index=False)
    scenario_stat.to_csv(report_dir / f'{name}_scenario_target_stat.csv', index=False)
    imp = pd.concat(importances, ignore_index=True)
    imp.to_csv(importance_dir / f'{name}_feature_importance_by_fold.csv', index=False)
    imp_mean = imp.groupby('feature', as_index=False)[['gain', 'split']].mean().sort_values('gain', ascending=False)
    imp_mean.to_csv(importance_dir / f'{name}_feature_importance_mean.csv', index=False)
    sub = pd.read_csv('sample_submission.csv')[['ID']]
    pred_frame = pd.DataFrame({'ID': test['ID'].values, fe_neighbor_feature_missing_exps_TARGET: np.clip(test_pred, 0, None)})
    sub = sub.merge(pred_frame, on='ID', how='left')
    sub[fe_neighbor_feature_missing_exps_TARGET] = sub[fe_neighbor_feature_missing_exps_TARGET].clip(lower=0)
    sub.to_csv(submission_dir / f'{name}_submission.csv', index=False)
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        import json
        json.dump({'hypothesis': hypothesis, 'fill_strategy': fill_strategy, 'feature_count': len(feature_cols), 'features': feature_cols, 'metadata': metadata}, f, ensure_ascii=False, indent=2)
    group_mae = float(summary.loc[summary['validation'] == 'groupkfold', 'mae'].iloc[0])
    target_heavy_holdout_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'mae'].iloc[0])
    high50_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high50_mae'].iloc[0])
    high100_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high100_mae'].iloc[0])
    fe_experiments_append_history(fe_neighbor_feature_missing_exps_OUTPUT_ROOT, {'experiment_name': name, 'base_after_run': 'neighbor_feature_missing', 'hypothesis': hypothesis, 'feature_count': len(feature_cols), 'groupkfold_mae': group_mae, 'target_heavy_holdout_mae': target_heavy_holdout_mae, 'high50_mae': high50_mae, 'high100_mae': high100_mae, 'improved_target_heavy_holdout': target_heavy_holdout_mae < fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT, 'best_target_heavy_holdout_after_run': min(target_heavy_holdout_mae, fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT), 'removed_count': 0})
    print(f'{name}: target_heavy_holdout={target_heavy_holdout_mae:.6f}, group={group_mae:.6f}, features={len(feature_cols)}', flush=True)
    return {'target_heavy_holdout_mae': target_heavy_holdout_mae, 'groupkfold_mae': group_mae, 'feature_cols': feature_cols}

def fe_neighbor_feature_missing_exps_run_default(name: str, hypothesis: str, train, test, cols, metadata):
    cfg = fe_experiments_ExperimentConfig(name=name, hypothesis=hypothesis, update_base=False)
    result = fe_experiments_run_experiment(cfg, train, test, cols, fe_neighbor_feature_missing_exps_OUTPUT_ROOT, metadata)
    fe_experiments_append_history(fe_neighbor_feature_missing_exps_OUTPUT_ROOT, {'experiment_name': name, 'base_after_run': 'neighbor_feature_missing', 'hypothesis': hypothesis, 'feature_count': result['feature_count'], 'groupkfold_mae': result['groupkfold_mae'], 'target_heavy_holdout_mae': result['target_heavy_holdout_mae'], 'high50_mae': result['high50_mae'], 'high100_mae': result['high100_mae'], 'improved_target_heavy_holdout': result['target_heavy_holdout_mae'] < fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT, 'best_target_heavy_holdout_after_run': min(result['target_heavy_holdout_mae'], fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT), 'removed_count': 0})
    print(f"{name}: target_heavy_holdout={result['target_heavy_holdout_mae']:.6f}, group={result['groupkfold_mae']:.6f}, features={result['feature_count']}", flush=True)
    return result

def fe_neighbor_feature_missing_exps_main() -> None:
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols = fe_neighbor_feature_missing_exps_best_columns(train, metadata)
    core = fe_neighbor_feature_missing_exps_add_neighbor_core_features(train, test)
    core_cols = list(dict.fromkeys(base_cols + core))
    core_result = fe_neighbor_feature_missing_exps_run_default('neighbor_core_features', 'Add 10 neighbor-analysis-derived pack/battery/robot-congestion features.', train, test, core_cols, {**metadata, 'added_features': core})
    best_cols = base_cols
    best_name = 'full_roll_best_features'
    best_target_heavy_holdout = fe_neighbor_feature_missing_exps_BEST_TARGET_HEAVY_HOLDOUT
    if core_result['target_heavy_holdout_mae'] < best_target_heavy_holdout:
        best_cols = core_cols
        best_name = 'neighbor_core_features'
        best_target_heavy_holdout = core_result['target_heavy_holdout_mae']
        extra = fe_neighbor_feature_missing_exps_add_neighbor_extra_features(train, test)
        extra_cols = list(dict.fromkeys(core_cols + extra))
        extra_result = fe_neighbor_feature_missing_exps_run_default('neighbor_extra_features', 'Add extra neighbor-analysis-derived agreement/count/compound pressure features after core improvement.', train, test, extra_cols, {**metadata, 'added_features': core + extra, 'base_after_core_target_heavy_holdout': best_target_heavy_holdout})
        if extra_result['target_heavy_holdout_mae'] < best_target_heavy_holdout:
            best_cols = extra_cols
            best_name = 'neighbor_extra_features'
            best_target_heavy_holdout = extra_result['target_heavy_holdout_mae']
    else:
        print('Core neighbor features did not improve target_heavy_holdout; skipping extra neighbor features.', flush=True)
    print(f'Missing-value experiments use best so far: {best_name} target_heavy_holdout={best_target_heavy_holdout:.6f}', flush=True)
    for strategy, desc in [('lag_roll_ffill_bfill', 'Fill lag/roll/diff by scenario ffill then bfill, others -999.'), ('lag_roll_linear_interpolate', 'Fill lag/roll/diff by scenario linear interpolation, others -999.'), ('roll_std_zero', 'Fill roll std missing values with 0, others -999.')]:
        fe_neighbor_feature_missing_exps_run_experiment_with_fill(name=f'missing_fill_{strategy}', hypothesis=desc, train=train, test=test, feature_cols=best_cols, metadata={**metadata, 'missing_base_experiment': best_name, 'missing_base_target_heavy_holdout': best_target_heavy_holdout}, fill_strategy=strategy)

# =============================================================================
# timeslot/late pressure 피처 실험
# =============================================================================

def fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test) -> list[str]:
    created = ['timeslot_ratio', 'remaining_slots', 'late_congestion', 'late_pack_pressure', 'late_robot_idle']
    for df in [train, test]:
        df['timeslot_ratio'] = df['timeslot'] / 24
        df['remaining_slots'] = 24 - df['timeslot']
        df['late_congestion'] = df['remaining_slots'] * df['congestion_score']
        df['late_pack_pressure'] = df['remaining_slots'] * df['pack_utilization']
        df['late_robot_idle'] = df['remaining_slots'] * df['robot_idle']
    return created

def fe_lgbm_timeslot_late_features_main() -> None:
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols = fe_neighbor_feature_missing_exps_best_columns(train, metadata)
    added = fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test)
    feature_cols = list(dict.fromkeys(base_cols + added))
    fe_neighbor_feature_missing_exps_run_experiment_with_fill(name='timeslot_late_features_lgbm', hypothesis='Add timeslot ratio, remaining slots, and remaining-slots weighted congestion/pack/idle features to the current public-LB best linear-interpolation LGBM feature set.', train=train, test=test, feature_cols=feature_cols, metadata={**metadata, 'base_experiment': 'missing_fill_lag_roll_linear_interpolate', 'base_public_lb': 10.0825226656, 'roll_windows': fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, 'lag_diff_steps': sorted(fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS), 'added_features': added, 'missing_indicator_features': []}, fill_strategy='lag_roll_linear_interpolate')

# =============================================================================
# scenario 단위 누적/초기 요약 피처 실험
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
fe_scenario_level_exps_OUTPUT_ROOT = Path('outputs')
fe_scenario_level_exps_BEST_EXP = 'timeslot_late_features_lgbm'
fe_scenario_level_exps_BEST_SUBMISSION = fe_scenario_level_exps_OUTPUT_ROOT / fe_scenario_level_exps_BEST_EXP / 'submissions' / f'{fe_scenario_level_exps_BEST_EXP}_submission.csv'
fe_scenario_level_exps_BEST_GROUP_OOF = fe_scenario_level_exps_OUTPUT_ROOT / fe_scenario_level_exps_BEST_EXP / 'oof_predictions' / f'{fe_scenario_level_exps_BEST_EXP}_groupkfold_oof.csv'

def fe_scenario_level_exps_shifted_group(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby('scenario_id', sort=False)[col].shift(1)

def fe_scenario_level_exps_expanding_mean(df: pd.DataFrame, shifted: pd.Series) -> pd.Series:
    return shifted.groupby(df['scenario_id'], sort=False).expanding().mean().reset_index(level=0, drop=True).sort_index()

def fe_scenario_level_exps_expanding_max(df: pd.DataFrame, shifted: pd.Series) -> pd.Series:
    return shifted.groupby(df['scenario_id'], sort=False).expanding().max().reset_index(level=0, drop=True).sort_index()

def fe_scenario_level_exps_add_cumulative_scenario_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    thresholds = {'congestion': float(train['congestion_score'].quantile(0.8)), 'pack': float(train['pack_utilization'].quantile(0.8)), 'order': float(train['order_inflow_15m'].quantile(0.8)), 'robot_shortage': float(train['robot_shortage_ratio'].quantile(0.8))}
    created = ['sc_congestion_mean_so_far', 'sc_congestion_max_so_far', 'sc_pack_mean_so_far', 'sc_backorder_max_so_far', 'sc_order_sum_so_far', 'sc_order_mean_so_far', 'sc_robot_shortage_mean_so_far', 'sc_risk_count_so_far', 'late16_sc_risk_count']
    for df in [train, test]:
        cong_shift = fe_scenario_level_exps_shifted_group(df, 'congestion_score')
        pack_shift = fe_scenario_level_exps_shifted_group(df, 'pack_utilization')
        back_shift = fe_scenario_level_exps_shifted_group(df, 'backorder_ratio')
        order_shift = fe_scenario_level_exps_shifted_group(df, 'order_inflow_15m')
        robot_short_shift = fe_scenario_level_exps_shifted_group(df, 'robot_shortage_ratio')
        df['sc_congestion_mean_so_far'] = fe_scenario_level_exps_expanding_mean(df, cong_shift)
        df['sc_congestion_max_so_far'] = fe_scenario_level_exps_expanding_max(df, cong_shift)
        df['sc_pack_mean_so_far'] = fe_scenario_level_exps_expanding_mean(df, pack_shift)
        df['sc_backorder_max_so_far'] = fe_scenario_level_exps_expanding_max(df, back_shift)
        df['sc_order_sum_so_far'] = order_shift.fillna(0).groupby(df['scenario_id'], sort=False).cumsum()
        df['sc_order_mean_so_far'] = fe_scenario_level_exps_expanding_mean(df, order_shift)
        df['sc_robot_shortage_mean_so_far'] = fe_scenario_level_exps_expanding_mean(df, robot_short_shift)
        df['sc_risk_count_so_far'] = (df['sc_congestion_max_so_far'] >= thresholds['congestion']).astype(int) + (df['sc_pack_mean_so_far'] >= thresholds['pack']).astype(int) + (df['sc_order_mean_so_far'] >= thresholds['order']).astype(int) + (df['sc_robot_shortage_mean_so_far'] >= thresholds['robot_shortage']).astype(int) + (df['sc_backorder_max_so_far'] > 0).astype(int)
        late16 = (df['timeslot'] >= 16).astype(int)
        df['late16_sc_risk_count'] = late16 * df['sc_risk_count_so_far']
    return created

def fe_scenario_level_exps_add_early_scenario_summary(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    specs = {'first4_congestion_mean': ('congestion_score', 'mean'), 'first4_pack_mean': ('pack_utilization', 'mean'), 'first4_order_mean': ('order_inflow_15m', 'mean'), 'first4_backorder_max': ('backorder_ratio', 'max'), 'first4_robot_shortage_mean': ('robot_shortage_ratio', 'mean')}
    created = list(specs.keys()) + [f'{name}_late16' for name in specs]
    for df in [train, test]:
        active = df['timeslot'] >= 4
        late16 = (df['timeslot'] >= 16).astype(int)
        for out_col, (src, how) in specs.items():
            if how == 'mean':
                values = df.groupby('scenario_id', sort=False)[src].transform(lambda s: s.iloc[:4].mean())
            elif how == 'max':
                values = df.groupby('scenario_id', sort=False)[src].transform(lambda s: s.iloc[:4].max())
            else:
                raise ValueError(how)
            df[out_col] = np.where(active, values, 0.0)
            df[f'{out_col}_late16'] = df[out_col] * late16
    return created

def fe_scenario_level_exps_base_feature_cols(train: pd.DataFrame, test: pd.DataFrame, metadata: dict) -> tuple[list[str], list[str]]:
    base_cols = fe_neighbor_feature_missing_exps_best_columns(train, metadata)
    late_base = fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test)
    return (list(dict.fromkeys(base_cols + late_base)), late_base)

def fe_scenario_level_exps_run_feature_experiments() -> None:
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols, late_base = fe_scenario_level_exps_base_feature_cols(train, test, metadata)
    cum_features = fe_scenario_level_exps_add_cumulative_scenario_features(train, test)
    fe_neighbor_feature_missing_exps_run_experiment_with_fill(name='scenario_cumulative_risk_lgbm', hypothesis='Add past-only cumulative scenario risk features to current public-LB best feature set.', train=train, test=test, feature_cols=list(dict.fromkeys(base_cols + cum_features)), metadata={**metadata, 'base_experiment': fe_scenario_level_exps_BEST_EXP, 'roll_windows': fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, 'lag_diff_steps': sorted(fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS), 'added_features': late_base + cum_features, 'scenario_feature_type': 'past_only_cumulative'}, fill_strategy='lag_roll_linear_interpolate')
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols, late_base = fe_scenario_level_exps_base_feature_cols(train, test, metadata)
    early_features = fe_scenario_level_exps_add_early_scenario_summary(train, test)
    fe_neighbor_feature_missing_exps_run_experiment_with_fill(name='scenario_first4_summary_lgbm', hypothesis='Add first-4-slot scenario summary features activated only at timeslot >= 4.', train=train, test=test, feature_cols=list(dict.fromkeys(base_cols + early_features)), metadata={**metadata, 'base_experiment': fe_scenario_level_exps_BEST_EXP, 'roll_windows': fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, 'lag_diff_steps': sorted(fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS), 'added_features': late_base + early_features, 'scenario_feature_type': 'first4_active_after_slot4'}, fill_strategy='lag_roll_linear_interpolate')

def fe_scenario_level_exps_normalize_risk(df: pd.DataFrame) -> pd.Series:
    risk = df['late16_sc_risk_count'].fillna(0).astype(float)
    max_risk = max(float(risk.max()), 1.0)
    return risk / max_risk

def fe_scenario_level_exps_run_postprocess_variants() -> None:
    out_dir = fe_scenario_level_exps_OUTPUT_ROOT / 'scenario_risk_postprocess'
    report_dir = out_dir / 'validation_reports'
    submission_dir = out_dir / 'submissions'
    report_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    train, test, _ = fe_experiments_build_feature_store()
    fe_scenario_level_exps_add_cumulative_scenario_features(train, test)
    train_risk = fe_scenario_level_exps_normalize_risk(train)
    test_risk = fe_scenario_level_exps_normalize_risk(test)
    oof = pd.read_csv(fe_scenario_level_exps_BEST_GROUP_OOF)
    sub = pd.read_csv(fe_scenario_level_exps_BEST_SUBMISSION)
    pred_col = fe_neighbor_feature_missing_exps_TARGET
    factors = [0.005, 0.01, 0.015, 0.02, 0.03]
    summary_rows = []
    bin_frames = []
    for factor in factors:
        name = f"scenario_risk_postprocess_x{str(factor).replace('.', 'p')}"
        adj_oof = oof.copy()
        adj_oof['pred'] = np.clip(adj_oof['pred'].values * (1.0 + factor * train_risk.values), 0, None)
        adj_oof['error'] = adj_oof['pred'] - adj_oof['target']
        adj_oof['abs_error'] = np.abs(adj_oof['error'])
        summary = val_linear_summarize_prediction_frame(adj_oof, 'groupkfold_postprocess', name)
        summary_rows.append(summary)
        bin_frames.append(val_linear_make_bin_report(adj_oof, 'groupkfold_postprocess', name))
        adj_sub = sub.copy()
        adj_sub[pred_col] = np.clip(adj_sub[pred_col].values * (1.0 + factor * test_risk.values), 0, None)
        adj_sub.to_csv(submission_dir / f'{name}_submission.csv', index=False)
    pd.concat(summary_rows, ignore_index=True).to_csv(report_dir / 'scenario_risk_postprocess_summary.csv', index=False)
    pd.concat(bin_frames, ignore_index=True).to_csv(report_dir / 'scenario_risk_postprocess_bin_report.csv', index=False)
    with open(out_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_experiment': fe_scenario_level_exps_BEST_EXP, 'base_submission': str(fe_scenario_level_exps_BEST_SUBMISSION), 'rule': 'pred *= 1 + factor * normalized late16 cumulative scenario risk', 'factors': factors}, f, ensure_ascii=False, indent=2)
    print(pd.concat(summary_rows, ignore_index=True).to_string(index=False), flush=True)

def fe_scenario_level_exps_main() -> None:
    fe_scenario_level_exps_run_feature_experiments()
    fe_scenario_level_exps_run_postprocess_variants()

# =============================================================================
# rolling window와 lag 조합 탐색
# =============================================================================

import argparse
import json
from itertools import combinations
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
fe_window_lag_search_ROLL_CANDIDATES = list(range(2, 26))
fe_window_lag_search_LAG_CANDIDATES = list(range(1, 11))
fe_window_lag_search_BASELINE_LAGS = [1, 2, 3, 5, 10]

def fe_window_lag_search_add_all_lag_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col in fe_experiments_LAG_COLS:
        group = df.groupby('scenario_id', sort=False)[col]
        lag_cache = {}
        for lag in fe_window_lag_search_LAG_CANDIDATES:
            lag_cache[lag] = group.shift(lag)
            new_cols[f'{col}_lag{lag}'] = lag_cache[lag]
        for lag in fe_window_lag_search_LAG_CANDIDATES:
            new_cols[f'{col}_diff{lag}'] = df[col] - lag_cache[lag]
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def fe_window_lag_search_build_search_feature_store() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    raw_feature_cols = [c for c in train.columns if c not in fe_experiments_RAW_ID_COLS + [fe_experiments_TARGET]]
    train, test = fe_experiments_add_layout_features(train, test, layout)
    out = []
    for df in [train, test]:
        fe_experiments_add_base_bottleneck_features(df)
        df.sort_values(['scenario_id', 'ID'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['timeslot'] = df.groupby('scenario_id', sort=False).cumcount()
        df = fe_window_lag_search_add_all_lag_diff_features(df)
        df = fe_experiments_add_rolling_features(df, windows=fe_window_lag_search_ROLL_CANDIDATES)
        out.append(df)
    metadata = {'raw_feature_cols': raw_feature_cols, 'roll_candidates': fe_window_lag_search_ROLL_CANDIDATES, 'lag_candidates': fe_window_lag_search_LAG_CANDIDATES, 'baseline_lags_for_rolling_search': fe_window_lag_search_BASELINE_LAGS}
    return (out[0], out[1], metadata)

def fe_window_lag_search_extract_roll_window(col: str) -> int | None:
    return fe_experiments_extract_roll_window(col)

def fe_window_lag_search_select_search_columns(train: pd.DataFrame, roll_windows: tuple[int, ...], lag_steps: tuple[int, ...]) -> list[str]:
    roll_set = set(roll_windows)
    lag_set = set(lag_steps)
    cols = []
    for col in train.columns:
        if col in fe_experiments_RAW_ID_COLS or col == fe_experiments_TARGET:
            continue
        if fe_experiments_is_roll_col(col):
            if fe_window_lag_search_extract_roll_window(col) in roll_set:
                cols.append(col)
            continue
        if fe_experiments_is_lag_col(col):
            step = int(col.rsplit('lag', 1)[1])
            if step in lag_set:
                cols.append(col)
            continue
        if fe_experiments_is_diff_col(col):
            step = int(col.rsplit('diff', 1)[1])
            if step in lag_set:
                cols.append(col)
            continue
        cols.append(col)
    return cols

def fe_window_lag_search_evaluate_target_heavy_holdout(train: pd.DataFrame, feature_cols: list[str], model_name: str, output_dir: Path | None=None) -> dict:
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(-999)
    y = train[fe_experiments_TARGET].astype(float)
    groups = train['scenario_id'].values
    config = val_linear_TargetHeavyConfig(random_state=42)
    train_mask, valid_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, config)
    model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
    model.fit(x.loc[train_mask], y.loc[train_mask], eval_set=[(x.loc[valid_mask], y.loc[valid_mask])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(model.predict(x.loc[valid_mask]), 0, None)
    mae = float(mean_absolute_error(y.loc[valid_mask], pred))
    pred_df = val_linear_make_prediction_frame(y.loc[valid_mask].values, pred, groups=groups[valid_mask])
    summary = val_linear_summarize_prediction_frame(pred_df, 'target_heavy_target_heavy_holdout', model_name)
    bin_report = val_linear_make_bin_report(pred_df, 'target_heavy_target_heavy_holdout', model_name)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / f'{model_name}_summary.csv', index=False)
        bin_report.to_csv(output_dir / f'{model_name}_bin_report.csv', index=False)
        pred_df.to_csv(output_dir / f'{model_name}_target_heavy_holdout_oof.csv', index=False)
        scenario_stat.to_csv(output_dir / f'{model_name}_scenario_target_stat.csv', index=False)
    return {'mae': mae, 'high50_mae': float(summary['high50_mae'].iloc[0]), 'high100_mae': float(summary['high100_mae'].iloc[0]), 'pred_max': float(summary['pred_max'].iloc[0]), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators']), 'feature_count': len(feature_cols)}

def fe_window_lag_search_combo_key(kind: str, rolls: tuple[int, ...], lags: tuple[int, ...]) -> tuple:
    return (kind, tuple(sorted(rolls)), tuple(sorted(lags)))

def fe_window_lag_search_load_history(path: Path) -> tuple[pd.DataFrame, set[tuple]]:
    if not path.exists():
        return (pd.DataFrame(), set())
    history = pd.read_csv(path)
    seen = set()
    for row in history.itertuples(index=False):
        rolls = tuple((int(x) for x in str(row.roll_windows).split() if x))
        lags = tuple((int(x) for x in str(row.lag_steps).split() if x))
        seen.add(fe_window_lag_search_combo_key(row.kind, rolls, lags))
    return (history, seen)

def fe_window_lag_search_append_row(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)

def fe_window_lag_search_evaluate_candidate(train: pd.DataFrame, history_path: Path, seen: set[tuple], kind: str, rolls: tuple[int, ...], lags: tuple[int, ...], eval_index: int) -> dict | None:
    rolls = tuple(sorted(rolls))
    lags = tuple(sorted(lags))
    key = fe_window_lag_search_combo_key(kind, rolls, lags)
    if key in seen:
        return None
    seen.add(key)
    model_name = f"{eval_index:04d}_{kind}_r{'-'.join(map(str, rolls))}_l{'-'.join(map(str, lags))}"
    feature_cols = fe_window_lag_search_select_search_columns(train, rolls, lags)
    result = fe_window_lag_search_evaluate_target_heavy_holdout(train, feature_cols, model_name)
    row = {'eval_index': eval_index, 'kind': kind, 'roll_windows': ' '.join(map(str, rolls)), 'lag_steps': ' '.join(map(str, lags)), 'n_roll': len(rolls), 'n_lag': len(lags), 'feature_count': result['feature_count'], 'target_heavy_holdout_mae': result['mae'], 'high50_mae': result['high50_mae'], 'high100_mae': result['high100_mae'], 'pred_max': result['pred_max'], 'best_iteration': result['best_iteration']}
    fe_window_lag_search_append_row(history_path, row)
    print(f"{eval_index:04d} {kind} rolls={rolls} lags={lags} target_heavy_holdout={result['mae']:.6f} features={result['feature_count']}", flush=True)
    return row

def fe_window_lag_search_top_rows(history_path: Path, kind_prefix: str | None=None, n: int=12) -> pd.DataFrame:
    history = pd.read_csv(history_path)
    if kind_prefix is not None:
        history = history[history['kind'].str.startswith(kind_prefix)]
    return history.sort_values('target_heavy_holdout_mae').head(n)

def fe_window_lag_search_parse_steps(value: str) -> tuple[int, ...]:
    return tuple((int(x) for x in str(value).split() if x))

def fe_window_lag_search_generate_neighbors(current: tuple[int, ...], candidates: list[int]) -> list[tuple[int, ...]]:
    current_set = set(current)
    neighbors = set()
    for c in candidates:
        if c not in current_set:
            neighbors.add(tuple(sorted(current_set | {c})))
    if len(current_set) > 1:
        for c in current:
            neighbors.add(tuple(sorted(current_set - {c})))
    for remove in current:
        for add in candidates:
            if add not in current_set:
                neighbors.add(tuple(sorted(current_set - {remove} | {add})))
    return sorted(neighbors, key=lambda x: (len(x), x))

def fe_window_lag_search_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', type=int, default=400)
    parser.add_argument('--rolling-budget', type=int, default=320)
    parser.add_argument('--beam-width', type=int, default=12)
    parser.add_argument('--seed-top', type=int, default=8)
    parser.add_argument('--output-dir', default='outputs/window_lag_search')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / 'history.csv'
    train, _, metadata = fe_window_lag_search_build_search_feature_store()
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    _, seen = fe_window_lag_search_load_history(history_path)
    eval_index = len(seen) + 1
    baseline_lags = tuple(fe_window_lag_search_BASELINE_LAGS)
    for roll in fe_window_lag_search_ROLL_CANDIDATES:
        if eval_index > args.max_evals:
            break
        row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'rolling_single', (roll,), baseline_lags, eval_index)
        if row is not None:
            eval_index += 1
    seed_rows = fe_window_lag_search_top_rows(history_path, 'rolling', args.seed_top)
    beam = [fe_window_lag_search_parse_steps(r.roll_windows) for r in seed_rows.itertuples(index=False)]
    for size in range(2, len(fe_window_lag_search_ROLL_CANDIDATES) + 1):
        if eval_index > min(args.rolling_budget, args.max_evals):
            break
        candidates = set()
        for combo in beam:
            for roll in fe_window_lag_search_ROLL_CANDIDATES:
                if roll not in combo:
                    nxt = tuple(sorted(set(combo) | {roll}))
                    if len(nxt) == size:
                        candidates.add(nxt)
        for rolls in sorted(candidates, key=lambda x: (sum((abs(a - b) for a, b in zip(x, x[1:]))), x)):
            if eval_index > min(args.rolling_budget, args.max_evals):
                break
            row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'rolling_beam', rolls, baseline_lags, eval_index)
            if row is not None:
                eval_index += 1
        beam_rows = fe_window_lag_search_top_rows(history_path, 'rolling', args.beam_width)
        beam = [fe_window_lag_search_parse_steps(r.roll_windows) for r in beam_rows.itertuples(index=False)]
    while eval_index <= min(args.rolling_budget, args.max_evals):
        best_rolls = fe_window_lag_search_parse_steps(fe_window_lag_search_top_rows(history_path, 'rolling', 1).iloc[0]['roll_windows'])
        progressed = False
        for rolls in fe_window_lag_search_generate_neighbors(best_rolls, fe_window_lag_search_ROLL_CANDIDATES):
            if eval_index > min(args.rolling_budget, args.max_evals):
                break
            row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'rolling_local', rolls, baseline_lags, eval_index)
            if row is not None:
                eval_index += 1
                progressed = True
        if not progressed:
            break
    best_rolls = fe_window_lag_search_parse_steps(fe_window_lag_search_top_rows(history_path, 'rolling', 1).iloc[0]['roll_windows'])
    print(f'Best rolling before lag search: {best_rolls}', flush=True)
    for lag in fe_window_lag_search_LAG_CANDIDATES:
        if eval_index > args.max_evals:
            break
        row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'lag_single', best_rolls, (lag,), eval_index)
        if row is not None:
            eval_index += 1
    lag_seed_rows = fe_window_lag_search_top_rows(history_path, 'lag', min(args.seed_top, 10))
    lag_beam = [fe_window_lag_search_parse_steps(r.lag_steps) for r in lag_seed_rows.itertuples(index=False)]
    for size in range(2, len(fe_window_lag_search_LAG_CANDIDATES) + 1):
        if eval_index > args.max_evals:
            break
        candidates = set()
        for combo in lag_beam:
            for lag in fe_window_lag_search_LAG_CANDIDATES:
                if lag not in combo:
                    nxt = tuple(sorted(set(combo) | {lag}))
                    if len(nxt) == size:
                        candidates.add(nxt)
        for lags in sorted(candidates, key=lambda x: (len(x), x)):
            if eval_index > args.max_evals:
                break
            row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'lag_beam', best_rolls, lags, eval_index)
            if row is not None:
                eval_index += 1
        lag_rows = fe_window_lag_search_top_rows(history_path, 'lag', args.beam_width)
        lag_beam = [fe_window_lag_search_parse_steps(r.lag_steps) for r in lag_rows.itertuples(index=False)]
    while eval_index <= args.max_evals:
        best_lags = fe_window_lag_search_parse_steps(fe_window_lag_search_top_rows(history_path, 'lag', 1).iloc[0]['lag_steps'])
        progressed = False
        for lags in fe_window_lag_search_generate_neighbors(best_lags, fe_window_lag_search_LAG_CANDIDATES):
            if eval_index > args.max_evals:
                break
            row = fe_window_lag_search_evaluate_candidate(train, history_path, seen, 'lag_local', best_rolls, lags, eval_index)
            if row is not None:
                eval_index += 1
                progressed = True
        if not progressed:
            break
    history = pd.read_csv(history_path).sort_values('target_heavy_holdout_mae')
    history.head(50).to_csv(output_dir / 'top50.csv', index=False)
    print('\nTop 10 candidates:')
    print(history.head(10).to_string(index=False))

# =============================================================================
# log-target용 rolling window 탐색
# =============================================================================

import argparse
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
fe_log_roll_window_search_OUTPUT_DIR = Path('outputs/log_roll_window_search')

def fe_log_roll_window_search_build_search_store() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    layout = pd.read_csv('data/raw/layout_info.csv')
    raw_feature_cols = [c for c in train.columns if c not in fe_experiments_RAW_ID_COLS + [fe_experiments_TARGET]]
    train, test = fe_experiments_add_layout_features(train, test, layout)
    out = []
    for df in [train, test]:
        fe_experiments_add_base_bottleneck_features(df)
        df.sort_values(['scenario_id', 'ID'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['timeslot'] = df.groupby('scenario_id', sort=False).cumcount()
        df = fe_window_lag_search_add_all_lag_diff_features(df)
        df = fe_experiments_add_rolling_features(df, windows=fe_window_lag_search_ROLL_CANDIDATES)
        out.append(df)
    train, test = out
    late_features = fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test)
    scenario_features = fe_scenario_level_exps_add_cumulative_scenario_features(train, test)
    metadata = {'experiment_family': 'log1p_roll_window_search', 'raw_feature_cols': raw_feature_cols, 'roll_candidates': fe_window_lag_search_ROLL_CANDIDATES, 'lag_candidates': fe_window_lag_search_LAG_CANDIDATES, 'fixed_lag_diff_steps': fe_window_lag_search_BASELINE_LAGS, 'target_transform': 'log1p target for training, expm1 for validation', 'fill_strategy': 'lag_roll_linear_interpolate', 'added_features': late_features + scenario_features}
    return (train, test, metadata)

def fe_log_roll_window_search_select_columns(train: pd.DataFrame, roll_windows: tuple[int, ...], lag_steps: tuple[int, ...]) -> list[str]:
    roll_set = set(roll_windows)
    lag_set = set(lag_steps)
    cols = []
    for col in train.columns:
        if col in fe_experiments_RAW_ID_COLS or col == fe_experiments_TARGET:
            continue
        if fe_experiments_is_roll_col(col):
            window = fe_experiments_extract_roll_window(col)
            if window in roll_set:
                cols.append(col)
            continue
        if fe_experiments_is_lag_col(col):
            step = int(col.rsplit('lag', 1)[1])
            if step in lag_set:
                cols.append(col)
            continue
        if fe_experiments_is_diff_col(col):
            step = int(col.rsplit('diff', 1)[1])
            if step in lag_set:
                cols.append(col)
            continue
        cols.append(col)
    return cols

def fe_log_roll_window_search_load_history(path: Path) -> tuple[pd.DataFrame, set[tuple]]:
    if not path.exists():
        return (pd.DataFrame(), set())
    history = pd.read_csv(path)
    seen = set()
    for row in history.itertuples(index=False):
        rolls = tuple((int(x) for x in str(row.roll_windows).split() if x))
        lags = tuple((int(x) for x in str(row.lag_steps).split() if x))
        seen.add(fe_window_lag_search_combo_key(row.kind, rolls, lags))
    return (history, seen)

def fe_log_roll_window_search_append_row(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)

def fe_log_roll_window_search_evaluate_target_heavy_holdout_log(train: pd.DataFrame, feature_cols: list[str], model_name: str) -> dict:
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    lag_roll_cols = [c for c in feature_cols if '_lag' in c or '_diff' in c or '_roll' in c]
    if lag_roll_cols:
        x[lag_roll_cols] = x[lag_roll_cols].groupby(train['scenario_id'], sort=False).transform(lambda s: s.interpolate(method='linear', limit_direction='both'))
    x = x.fillna(-999)
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    y_log = np.log1p(y)
    groups = train['scenario_id'].values
    tr_mask, va_mask, _ = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, val_linear_TargetHeavyConfig(random_state=42))
    model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
    model.fit(x.loc[tr_mask], y_log.loc[tr_mask], eval_set=[(x.loc[va_mask], y_log.loc[va_mask])], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = np.clip(np.expm1(model.predict(x.loc[va_mask])), 0, None)
    mae = float(mean_absolute_error(y.loc[va_mask], pred))
    pred_df = val_linear_make_prediction_frame(y.loc[va_mask].values, pred, groups=groups[va_mask])
    summary = val_linear_summarize_prediction_frame(pred_df, 'target_heavy_target_heavy_holdout', model_name)
    bin_report = val_linear_make_bin_report(pred_df, 'target_heavy_target_heavy_holdout', model_name)
    return {'mae': mae, 'high50_mae': float(summary['high50_mae'].iloc[0]), 'high100_mae': float(summary['high100_mae'].iloc[0]), 'pred_max': float(summary['pred_max'].iloc[0]), 'pred_mean': float(summary['pred_mean'].iloc[0]), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators']), 'feature_count': len(feature_cols), 'bin_report': bin_report}

def fe_log_roll_window_search_evaluate_candidate(train: pd.DataFrame, history_path: Path, seen: set[tuple], kind: str, rolls: tuple[int, ...], eval_index: int) -> dict | None:
    rolls = tuple(sorted(rolls))
    lags = tuple(fe_window_lag_search_BASELINE_LAGS)
    key = fe_window_lag_search_combo_key(kind, rolls, lags)
    if key in seen:
        return None
    seen.add(key)
    model_name = f"{eval_index:04d}_{kind}_r{'-'.join(map(str, rolls))}_l{'-'.join(map(str, lags))}"
    feature_cols = fe_log_roll_window_search_select_columns(train, rolls, lags)
    result = fe_log_roll_window_search_evaluate_target_heavy_holdout_log(train, feature_cols, model_name)
    row = {'eval_index': eval_index, 'kind': kind, 'roll_windows': ' '.join(map(str, rolls)), 'lag_steps': ' '.join(map(str, lags)), 'n_roll': len(rolls), 'n_lag': len(lags), 'feature_count': result['feature_count'], 'target_heavy_holdout_mae': result['mae'], 'high50_mae': result['high50_mae'], 'high100_mae': result['high100_mae'], 'pred_mean': result['pred_mean'], 'pred_max': result['pred_max'], 'best_iteration': result['best_iteration']}
    fe_log_roll_window_search_append_row(history_path, row)
    print(f"{eval_index:04d} {kind} rolls={rolls} target_heavy_holdout={result['mae']:.6f} high50={result['high50_mae']:.6f} high100={result['high100_mae']:.6f}", flush=True)
    return row

def fe_log_roll_window_search_top_rows(history_path: Path, kind_prefix: str | None=None, n: int=12) -> pd.DataFrame:
    history = pd.read_csv(history_path)
    if kind_prefix is not None:
        history = history[history['kind'].str.startswith(kind_prefix)]
    return history.sort_values('target_heavy_holdout_mae').head(n)

def fe_log_roll_window_search_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-evals', type=int, default=200)
    parser.add_argument('--beam-width', type=int, default=12)
    parser.add_argument('--seed-top', type=int, default=8)
    parser.add_argument('--output-dir', default=str(fe_log_roll_window_search_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / 'history.csv'
    train, _, metadata = fe_log_roll_window_search_build_search_store()
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    _, seen = fe_log_roll_window_search_load_history(history_path)
    eval_index = len(seen) + 1
    for roll in fe_window_lag_search_ROLL_CANDIDATES:
        if eval_index > args.max_evals:
            break
        row = fe_log_roll_window_search_evaluate_candidate(train, history_path, seen, 'rolling_single', (roll,), eval_index)
        if row is not None:
            eval_index += 1
    seed_rows = fe_log_roll_window_search_top_rows(history_path, 'rolling', args.seed_top)
    beam = [fe_window_lag_search_parse_steps(r.roll_windows) for r in seed_rows.itertuples(index=False)]
    for size in range(2, len(fe_window_lag_search_ROLL_CANDIDATES) + 1):
        if eval_index > args.max_evals:
            break
        candidates = set()
        for combo in beam:
            for roll in fe_window_lag_search_ROLL_CANDIDATES:
                if roll not in combo:
                    nxt = tuple(sorted(set(combo) | {roll}))
                    if len(nxt) == size:
                        candidates.add(nxt)
        for rolls in sorted(candidates, key=lambda x: (sum((abs(a - b) for a, b in zip(x, x[1:]))), x)):
            if eval_index > args.max_evals:
                break
            row = fe_log_roll_window_search_evaluate_candidate(train, history_path, seen, 'rolling_beam', rolls, eval_index)
            if row is not None:
                eval_index += 1
        beam_rows = fe_log_roll_window_search_top_rows(history_path, 'rolling', args.beam_width)
        beam = [fe_window_lag_search_parse_steps(r.roll_windows) for r in beam_rows.itertuples(index=False)]
    while eval_index <= args.max_evals:
        best_rolls = fe_window_lag_search_parse_steps(fe_log_roll_window_search_top_rows(history_path, 'rolling', 1).iloc[0]['roll_windows'])
        progressed = False
        for rolls in fe_window_lag_search_generate_neighbors(best_rolls, fe_window_lag_search_ROLL_CANDIDATES):
            if eval_index > args.max_evals:
                break
            row = fe_log_roll_window_search_evaluate_candidate(train, history_path, seen, 'rolling_local', rolls, eval_index)
            if row is not None:
                eval_index += 1
                progressed = True
        if not progressed:
            break
    history = pd.read_csv(history_path).sort_values('target_heavy_holdout_mae')
    history.head(50).to_csv(output_dir / 'top50.csv', index=False)
    print('\nTop 10 candidates:')
    print(history.head(10).to_string(index=False), flush=True)

# =============================================================================
# LightGBM log-target 회귀 실험
# =============================================================================

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
fe_lgbm_log_target_exps_OUTPUT_ROOT = Path('outputs')

def fe_lgbm_log_target_exps_fit_log_lgbm(x_tr, y_tr_log, x_va, y_va_log):
    model = lgb.LGBMRegressor(**fe_experiments_LGBM_PARAMS)
    model.fit(x_tr, y_tr_log, eval_set=[(x_va, y_va_log)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

def fe_lgbm_log_target_exps_predict_target(model, x) -> np.ndarray:
    return np.clip(np.expm1(model.predict(x)), 0, None)

def fe_lgbm_log_target_exps_run_log_target_experiment(name: str, hypothesis: str, train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], metadata: dict) -> dict:
    output_dir = fe_lgbm_log_target_exps_OUTPUT_ROOT / name
    report_dir = output_dir / 'validation_reports'
    submission_dir = output_dir / 'submissions'
    oof_dir = output_dir / 'oof_predictions'
    importance_dir = output_dir / 'feature_importance'
    for path in [report_dir, submission_dir, oof_dir, importance_dir]:
        path.mkdir(parents=True, exist_ok=True)
    x, xt = fe_neighbor_feature_missing_exps_fill_features(train, test, feature_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    y_log = np.log1p(y)
    groups = train['scenario_id'].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(x))
    test_pred = np.zeros(len(xt))
    fold_rows = []
    importances = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = fe_lgbm_log_target_exps_fit_log_lgbm(x.iloc[tr_idx], y_log.iloc[tr_idx], x.iloc[va_idx], y_log.iloc[va_idx])
        pred = fe_lgbm_log_target_exps_predict_target(model, x.iloc[va_idx])
        oof[va_idx] = pred
        test_pred += fe_lgbm_log_target_exps_predict_target(model, xt) / 5
        fold_rows.append({'model_name': name, 'validation': 'groupkfold', 'fold': fold, 'mae': float(mean_absolute_error(y.iloc[va_idx], pred)), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators'])})
        importances.append(pd.DataFrame({'model_name': name, 'feature': feature_cols, 'gain': model.booster_.feature_importance(importance_type='gain'), 'split': model.booster_.feature_importance(importance_type='split'), 'fold': fold}))
    group_oof = val_linear_make_prediction_frame(y.values, oof, groups=groups)
    group_summary = val_linear_summarize_prediction_frame(group_oof, 'groupkfold', name)
    group_bin = val_linear_make_bin_report(group_oof, 'groupkfold', name)
    tr_mask, va_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, val_linear_TargetHeavyConfig(random_state=42))
    exp_model = fe_lgbm_log_target_exps_fit_log_lgbm(x.loc[tr_mask], y_log.loc[tr_mask], x.loc[va_mask], y_log.loc[va_mask])
    exp_pred = fe_lgbm_log_target_exps_predict_target(exp_model, x.loc[va_mask])
    exp_oof = val_linear_make_prediction_frame(y.loc[va_mask].values, exp_pred, groups=groups[va_mask])
    exp_summary = val_linear_summarize_prediction_frame(exp_oof, 'target_heavy_target_heavy_holdout', name)
    exp_bin = val_linear_make_bin_report(exp_oof, 'target_heavy_target_heavy_holdout', name)
    summary = pd.concat([group_summary, exp_summary], ignore_index=True)
    bin_report = pd.concat([group_bin, exp_bin], ignore_index=True)
    fold_report = pd.DataFrame(fold_rows)
    summary.to_csv(report_dir / f'{name}_summary.csv', index=False)
    fold_report.to_csv(report_dir / f'{name}_fold_report.csv', index=False)
    bin_report.to_csv(report_dir / f'{name}_bin_report.csv', index=False)
    group_oof.to_csv(oof_dir / f'{name}_groupkfold_oof.csv', index=False)
    exp_oof.to_csv(oof_dir / f'{name}_target_heavy_target_heavy_holdout_oof.csv', index=False)
    scenario_stat.to_csv(report_dir / f'{name}_scenario_target_stat.csv', index=False)
    imp = pd.concat(importances, ignore_index=True)
    imp.to_csv(importance_dir / f'{name}_feature_importance_by_fold.csv', index=False)
    imp.groupby('feature', as_index=False)[['gain', 'split']].mean().sort_values('gain', ascending=False).to_csv(importance_dir / f'{name}_feature_importance_mean.csv', index=False)
    sub = pd.read_csv('sample_submission.csv')[['ID']]
    pred_frame = pd.DataFrame({'ID': test['ID'].values, fe_neighbor_feature_missing_exps_TARGET: np.clip(test_pred, 0, None)})
    sub = sub.merge(pred_frame, on='ID', how='left')
    sub[fe_neighbor_feature_missing_exps_TARGET] = sub[fe_neighbor_feature_missing_exps_TARGET].clip(lower=0)
    sub.to_csv(submission_dir / f'{name}_submission.csv', index=False)
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'hypothesis': hypothesis, 'target_transform': 'log1p target for training, expm1 predictions for scoring/submission', 'fill_strategy': 'lag_roll_linear_interpolate', 'feature_count': len(feature_cols), 'features': feature_cols, 'metadata': metadata}, f, ensure_ascii=False, indent=2)
    group_mae = float(summary.loc[summary['validation'] == 'groupkfold', 'mae'].iloc[0])
    target_heavy_holdout_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'mae'].iloc[0])
    high50_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high50_mae'].iloc[0])
    high100_mae = float(summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout', 'high100_mae'].iloc[0])
    fe_experiments_append_history(fe_lgbm_log_target_exps_OUTPUT_ROOT, {'experiment_name': name, 'base_after_run': 'log_target', 'hypothesis': hypothesis, 'feature_count': len(feature_cols), 'groupkfold_mae': group_mae, 'target_heavy_holdout_mae': target_heavy_holdout_mae, 'high50_mae': high50_mae, 'high100_mae': high100_mae, 'improved_target_heavy_holdout': target_heavy_holdout_mae < 10.7680876488716, 'best_target_heavy_holdout_after_run': min(target_heavy_holdout_mae, 10.7680876488716), 'removed_count': 0})
    print(f'{name}: target_heavy_holdout={target_heavy_holdout_mae:.6f}, group={group_mae:.6f}, features={len(feature_cols)}', flush=True)
    return {'target_heavy_holdout_mae': target_heavy_holdout_mae, 'groupkfold_mae': group_mae}

def fe_lgbm_log_target_exps_main() -> None:
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols = fe_neighbor_feature_missing_exps_best_columns(train, metadata)
    late_features = fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test)
    timeslot_late_feature_cols = list(dict.fromkeys(base_cols + late_features))
    fe_lgbm_log_target_exps_run_log_target_experiment(name='log_target_timeslot_late_features', hypothesis='현재 public LB 기준으로 좋았던 timeslot/late 피처셋으로 log1p(target) 모델을 학습합니다.', train=train, test=test, feature_cols=timeslot_late_feature_cols, metadata={**metadata, 'base_experiment': 'timeslot_late_features_lgbm', 'base_public_lb': 10.0813780133, 'roll_windows': fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, 'lag_diff_steps': sorted(fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS), 'added_features': late_features})
    train, test, metadata = fe_experiments_build_feature_store()
    base_cols = fe_neighbor_feature_missing_exps_best_columns(train, metadata)
    late_features = fe_lgbm_timeslot_late_features_add_timeslot_late_features(train, test)
    scenario_features = fe_scenario_level_exps_add_cumulative_scenario_features(train, test)
    scenario_cumulative_feature_cols = list(dict.fromkeys(base_cols + late_features + scenario_features))
    fe_lgbm_log_target_exps_run_log_target_experiment(name='log_target_scenario_cumulative_risk', hypothesis='scenario 누적 risk 피처셋으로 log1p(target) 모델을 학습합니다.', train=train, test=test, feature_cols=scenario_cumulative_feature_cols, metadata={**metadata, 'base_experiment': 'scenario_cumulative_risk_lgbm', 'roll_windows': fe_neighbor_feature_missing_exps_BEST_ROLL_WINDOWS, 'lag_diff_steps': sorted(fe_neighbor_feature_missing_exps_LAG_DIFF_STEPS), 'added_features': late_features + scenario_features})

# =============================================================================
# target<1 저지연 확률 피처 생성
# =============================================================================

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
fe_target_lt1_probability_OUTPUT_DIR = Path('outputs/target_lt1_probability')
fe_target_lt1_probability_BASE_LOG_TARGET_MODEL = 'log_target_selected_roll_window_lgbm'
fe_target_lt1_probability_ROLLS = (19, 20)
fe_target_lt1_probability_LAGS = (1, 2, 3, 5, 10)
fe_target_lt1_probability_CLF_PARAMS = {'objective': 'binary', 'learning_rate': 0.03, 'n_estimators': 1200, 'num_leaves': 63, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 40, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}

def fe_target_lt1_probability_make_low_target_prob_features(train: pd.DataFrame, test: pd.DataFrame, x: pd.DataFrame, xt: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    label = (y < 1).astype(int).values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(x), dtype=float)
    test_prob = np.zeros(len(xt), dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, label, groups), start=1):
        model = lgb.LGBMClassifier(**fe_target_lt1_probability_CLF_PARAMS)
        model.fit(x.iloc[tr_idx], label[tr_idx], eval_set=[(x.iloc[va_idx], label[va_idx])], eval_metric='auc', callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict_proba(x.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        test_prob += model.predict_proba(xt)[:, 1] / 5.0
        fold_rows.append({'fold': fold, 'roc_auc': float(roc_auc_score(label[va_idx], pred)), 'pr_auc': float(average_precision_score(label[va_idx], pred)), 'positive_rate': float(label[va_idx].mean()), 'best_iteration': int(model.best_iteration_ or fe_target_lt1_probability_CLF_PARAMS['n_estimators'])})
        print(f"low<1 clf fold={fold} roc_auc={fold_rows[-1]['roc_auc']:.6f} pr_auc={fold_rows[-1]['pr_auc']:.6f}", flush=True)
    created = ['lt1_clf_prob_oof', 'lt1_clf_prob_sq', 'lt1_clf_prob_ge_0p8', 'lt1_clf_prob_ge_0p9', 'lt1_clf_prob_ge_0p95']
    train = train.copy()
    test = test.copy()
    train['lt1_clf_prob_oof'] = oof
    test['lt1_clf_prob_oof'] = test_prob
    for df in [train, test]:
        df['lt1_clf_prob_sq'] = df['lt1_clf_prob_oof'] ** 2
        df['lt1_clf_prob_ge_0p8'] = (df['lt1_clf_prob_oof'] >= 0.8).astype(int)
        df['lt1_clf_prob_ge_0p9'] = (df['lt1_clf_prob_oof'] >= 0.9).astype(int)
        df['lt1_clf_prob_ge_0p95'] = (df['lt1_clf_prob_oof'] >= 0.95).astype(int)
    fe_target_lt1_probability_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'ID': train['ID'].values, fe_neighbor_feature_missing_exps_TARGET: y.values, 'lt1_clf_prob_oof': oof}).to_csv(fe_target_lt1_probability_OUTPUT_DIR / 'oof_prob_target_lt1.csv', index=False)
    pd.DataFrame({'ID': test['ID'].values, 'lt1_clf_prob': test_prob}).to_csv(fe_target_lt1_probability_OUTPUT_DIR / 'test_prob_target_lt1.csv', index=False)
    fold_report = pd.DataFrame(fold_rows)
    fold_report.to_csv(fe_target_lt1_probability_OUTPUT_DIR / 'classifier_fold_metrics.csv', index=False)
    clf_metrics = {'oof_roc_auc': float(roc_auc_score(label, oof)), 'oof_pr_auc': float(average_precision_score(label, oof)), 'positive_rate': float(label.mean()), 'folds': fold_rows}
    pd.DataFrame([clf_metrics | {'folds': json.dumps(fold_rows)}]).to_csv(fe_target_lt1_probability_OUTPUT_DIR / 'classifier_metrics.csv', index=False)
    return (train, test, created, clf_metrics)

def fe_target_lt1_probability_main() -> None:
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(train, fe_target_lt1_probability_ROLLS, fe_target_lt1_probability_LAGS)
    x, xt = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_prob, test_prob, prob_features, clf_metrics = fe_target_lt1_probability_make_low_target_prob_features(train, test, x, xt, y, groups)
    experiments = [('target_lt1_probability_feature_lgbm', prob_features[:1], 'Add OOF target<1 classifier probability as a regression feature.'), ('target_lt1_probability_features_lgbm', prob_features, 'Add OOF target<1 classifier probability, squared probability, and high-probability flags.')]
    rows = []
    for name, new_cols, hypothesis in experiments:
        result = fe_lgbm_log_target_exps_run_log_target_experiment(name=name, hypothesis=hypothesis, train=train_prob, test=test_prob, feature_cols=list(dict.fromkeys(base_cols + new_cols)), metadata={**metadata, 'base_feature_set': fe_target_lt1_probability_BASE_LOG_TARGET_MODEL, 'stacking_signal': 'OOF LGBMClassifier probability for target < 1', 'classifier_metrics': clf_metrics, 'new_features': new_cols, 'base_feature_count': len(base_cols)})
        rows.append({'experiment': name, 'feature_count': len(base_cols) + len(new_cols), 'new_feature_count': len(new_cols), **result, 'submission_path': str(Path('outputs') / name / 'submissions' / f'{name}_submission.csv')})
        pd.DataFrame(rows).sort_values(['target_heavy_holdout_mae', 'groupkfold_mae']).to_csv(fe_target_lt1_probability_OUTPUT_DIR / 'lt1_prob_regression_summary.csv', index=False)
    with open(fe_target_lt1_probability_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_feature_set': fe_target_lt1_probability_BASE_LOG_TARGET_MODEL, 'roll_windows': fe_target_lt1_probability_ROLLS, 'lag_diff_steps': fe_target_lt1_probability_LAGS, 'prob_features': prob_features, 'classifier_metrics': clf_metrics}, f, ensure_ascii=False, indent=2, default=str)
    print(pd.DataFrame(rows).sort_values(['target_heavy_holdout_mae', 'groupkfold_mae']).to_string(index=False), flush=True)

# =============================================================================
# delay-bin 분류 확률 피처 생성
# =============================================================================

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
fe_delay_bin_probability_features_OUTPUT_DIR = Path('outputs/delay_bin_probability_features')
fe_delay_bin_probability_features_BASE_EXP = 'target_lt1_probability_features_lgbm'
fe_delay_bin_probability_features_ROLLS = (19, 20)
fe_delay_bin_probability_features_LAGS = (1, 2, 3, 5, 10)
fe_delay_bin_probability_features_CLF_PARAMS = {'objective': 'multiclass', 'learning_rate': 0.03, 'n_estimators': 1200, 'num_leaves': 63, 'subsample': 0.85, 'colsample_bytree': 0.85, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 40, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}
fe_delay_bin_probability_features_BIN_SPECS = {'wide_tail': [0, 10, 20, 30, 40, 50, 70, 100, 150, 300, np.inf], 'stable': [0, 10, 20, 30, 40, 50, 100, np.inf]}

def fe_delay_bin_probability_features_bin_labels(edges: list[float]) -> list[str]:
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_s = str(int(lo)) if np.isfinite(lo) else 'min'
        hi_s = str(int(hi)) if np.isfinite(hi) else 'inf'
        labels.append(f'bin_{lo_s}_{hi_s}')
    return labels

def fe_delay_bin_probability_features_make_bin_target(y: pd.Series, edges: list[float]) -> np.ndarray:
    values = y.astype(float).values
    values = np.clip(values, edges[0], None)
    return np.digitize(values, edges[1:-1], right=False).astype(np.int16)

def fe_delay_bin_probability_features_aligned_proba(model: lgb.LGBMClassifier, x: pd.DataFrame, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(x)
    if proba.shape[1] == n_classes:
        return proba
    out = np.zeros((len(x), n_classes), dtype=float)
    for i, cls in enumerate(model.classes_):
        out[:, int(cls)] = proba[:, i]
    return out

def fe_delay_bin_probability_features_make_delay_bin_prob_features(x: pd.DataFrame, xt: pd.DataFrame, y: pd.Series, groups: np.ndarray, edges: list[float], spec_name: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    y_bin = fe_delay_bin_probability_features_make_bin_target(y, edges)
    labels = fe_delay_bin_probability_features_bin_labels(edges)
    n_classes = len(labels)
    params = dict(fe_delay_bin_probability_features_CLF_PARAMS)
    params['num_class'] = n_classes
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros((len(x), n_classes), dtype=float)
    test_prob = np.zeros((len(xt), n_classes), dtype=float)
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y_bin, groups), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(x.iloc[tr_idx], y_bin[tr_idx], eval_set=[(x.iloc[va_idx], y_bin[va_idx])], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = fe_delay_bin_probability_features_aligned_proba(model, x.iloc[va_idx], n_classes)
        oof[va_idx] = pred
        test_prob += fe_delay_bin_probability_features_aligned_proba(model, xt, n_classes) / 5.0
        fold_rows.append({'spec': spec_name, 'fold': fold, 'accuracy': float(accuracy_score(y_bin[va_idx], pred.argmax(axis=1))), 'log_loss': float(log_loss(y_bin[va_idx], np.clip(pred, 1e-07, 1 - 1e-07), labels=list(range(n_classes)))), 'best_iteration': int(model.best_iteration_ or params['n_estimators'])})
        print(f"{spec_name} fold={fold} acc={fold_rows[-1]['accuracy']:.6f} logloss={fold_rows[-1]['log_loss']:.6f}", flush=True)
    prob_cols = [f'delay_{spec_name}_prob_{label}' for label in labels]
    train_prob = pd.DataFrame(oof, columns=prob_cols, index=x.index)
    test_prob_df = pd.DataFrame(test_prob, columns=prob_cols, index=xt.index)
    mids = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isfinite(hi):
            mids.append((lo + hi) / 2.0)
        else:
            mids.append(lo)
    mids_arr = np.array(mids, dtype=float)
    train_prob[f'delay_{spec_name}_prob_expected_bin_mid'] = oof @ mids_arr
    test_prob_df[f'delay_{spec_name}_prob_expected_bin_mid'] = test_prob @ mids_arr
    train_prob[f'delay_{spec_name}_prob_high_ge50'] = oof[:, [i for i, lo in enumerate(edges[:-1]) if lo >= 50]].sum(axis=1)
    test_prob_df[f'delay_{spec_name}_prob_high_ge50'] = test_prob[:, [i for i, lo in enumerate(edges[:-1]) if lo >= 50]].sum(axis=1)
    train_prob[f'delay_{spec_name}_prob_high_ge100'] = oof[:, [i for i, lo in enumerate(edges[:-1]) if lo >= 100]].sum(axis=1)
    test_prob_df[f'delay_{spec_name}_prob_high_ge100'] = test_prob[:, [i for i, lo in enumerate(edges[:-1]) if lo >= 100]].sum(axis=1)
    created = list(train_prob.columns)
    bin_counts = pd.Series(y_bin).value_counts().sort_index()
    metrics = {'spec': spec_name, 'edges': [None if not np.isfinite(v) else float(v) for v in edges], 'labels': labels, 'created_features': created, 'folds': fold_rows, 'mean_accuracy': float(np.mean([r['accuracy'] for r in fold_rows])), 'mean_log_loss': float(np.mean([r['log_loss'] for r in fold_rows])), 'bin_counts': {labels[i]: int(bin_counts.get(i, 0)) for i in range(n_classes)}, 'bin_rates': {labels[i]: float(bin_counts.get(i, 0) / len(y_bin)) for i in range(n_classes)}}
    return (train_prob, test_prob_df, created, metrics)

def fe_delay_bin_probability_features_main() -> None:
    fe_delay_bin_probability_features_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(train, fe_delay_bin_probability_features_ROLLS, fe_delay_bin_probability_features_LAGS)
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_target_lt1, test_target_lt1, lt1_cols, lt1_metrics = fe_target_lt1_probability_make_low_target_prob_features(train, test, x_base, xt_base, y, groups)
    target_lt1_feature_cols = list(dict.fromkeys(base_cols + lt1_cols))
    x_target_lt1, xt_target_lt1 = fe_neighbor_feature_missing_exps_fill_features(train_target_lt1, test_target_lt1, target_lt1_feature_cols, 'lag_roll_linear_interpolate')
    rows = []
    all_metrics = {}
    fold_metric_rows = []
    for spec_name, edges in fe_delay_bin_probability_features_BIN_SPECS.items():
        tr_prob, te_prob, prob_cols, clf_metrics = fe_delay_bin_probability_features_make_delay_bin_prob_features(x=x_target_lt1, xt=xt_target_lt1, y=y, groups=groups, edges=edges, spec_name=spec_name)
        all_metrics[spec_name] = clf_metrics
        fold_metric_rows.extend(clf_metrics['folds'])
        train_run = pd.concat([train_target_lt1.reset_index(drop=True), tr_prob.reset_index(drop=True)], axis=1)
        test_run = pd.concat([test_target_lt1.reset_index(drop=True), te_prob.reset_index(drop=True)], axis=1)
        name = f'delay_bin_probability_{spec_name}'
        result = fe_lgbm_log_target_exps_run_log_target_experiment(name=name, hypothesis=f'Add OOF delay-bin classifier probabilities ({spec_name}) to LB-best target_lt1_features log-target LGBM regression features.', train=train_run, test=test_run, feature_cols=list(dict.fromkeys(target_lt1_feature_cols + prob_cols)), metadata={**metadata, 'base_feature_set': fe_delay_bin_probability_features_BASE_EXP, 'base_public_lb': 10.0124575581, 'lt1_classifier_metrics': lt1_metrics, 'delay_bin_classifier_metrics': clf_metrics, 'delay_bin_spec': spec_name, 'delay_bin_edges': clf_metrics['edges'], 'delay_bin_features': prob_cols})
        rows.append({'experiment': name, 'spec': spec_name, 'feature_count': len(target_lt1_feature_cols) + len(prob_cols), 'new_delay_bin_feature_count': len(prob_cols), 'clf_mean_accuracy': clf_metrics['mean_accuracy'], 'clf_mean_log_loss': clf_metrics['mean_log_loss'], **result, 'submission_path': str(Path('outputs') / name / 'submissions' / f'{name}_submission.csv')})
        pd.DataFrame(rows).sort_values(['target_heavy_holdout_mae', 'groupkfold_mae']).to_csv(fe_delay_bin_probability_features_OUTPUT_DIR / 'delay_bin_prob_feature_summary.csv', index=False)
    pd.DataFrame(fold_metric_rows).to_csv(fe_delay_bin_probability_features_OUTPUT_DIR / 'delay_bin_classifier_fold_metrics.csv', index=False)
    with open(fe_delay_bin_probability_features_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_feature_set': fe_delay_bin_probability_features_BASE_EXP, 'roll_windows': fe_delay_bin_probability_features_ROLLS, 'lag_diff_steps': fe_delay_bin_probability_features_LAGS, 'lt1_features': lt1_cols, 'lt1_classifier_metrics': lt1_metrics, 'delay_bin_classifier_metrics': all_metrics}, f, ensure_ascii=False, indent=2, default=str)
    print(pd.DataFrame(rows).sort_values(['target_heavy_holdout_mae', 'groupkfold_mae']).to_string(index=False), flush=True)

# =============================================================================
# wide-tail quantile LightGBM 모델
# =============================================================================

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
fe_wide_tail_quantile_lgbm_OUTPUT_DIR = Path('outputs/wide_tail_quantile_lgbm')
fe_wide_tail_quantile_lgbm_ROLLS = (19, 20)
fe_wide_tail_quantile_lgbm_LAGS = (1, 2, 3, 5, 10)
fe_wide_tail_quantile_lgbm_ALPHAS = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
fe_wide_tail_quantile_lgbm_BASE_PUBLIC_LB = 10.0124575581

def fe_wide_tail_quantile_lgbm_make_params(alpha: float) -> dict:
    params = dict(fe_experiments_LGBM_PARAMS)
    params.update({'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile', 'random_state': 42, 'verbosity': -1})
    return params

def fe_wide_tail_quantile_lgbm_fit_quantile_lgbm(x_tr, y_tr, x_va, y_va, alpha: float):
    model = lgb.LGBMRegressor(**fe_wide_tail_quantile_lgbm_make_params(alpha))
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model

def fe_wide_tail_quantile_lgbm_predict_target(model, x) -> np.ndarray:
    return np.clip(np.asarray(model.predict(x), dtype=float), 0, None)

def fe_wide_tail_quantile_lgbm_load_or_make_lt1_features(train: pd.DataFrame, test: pd.DataFrame, x_base: pd.DataFrame, xt_base: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    oof_path = Path('outputs/target_lt1_probability/oof_prob_target_lt1.csv')
    test_path = Path('outputs/target_lt1_probability/test_prob_target_lt1.csv')
    if not (oof_path.exists() and test_path.exists()):
        return fe_target_lt1_probability_make_low_target_prob_features(train, test, x_base, xt_base, y, groups)
    oof = pd.read_csv(oof_path)[['ID', 'lt1_clf_prob_oof']]
    te = pd.read_csv(test_path)[['ID', 'lt1_clf_prob']].rename(columns={'lt1_clf_prob': 'lt1_clf_prob_oof'})
    train_out = train.merge(oof, on='ID', how='left')
    test_out = test.merge(te, on='ID', how='left')
    for df in [train_out, test_out]:
        df['lt1_clf_prob_oof'] = df['lt1_clf_prob_oof'].fillna(0)
        df['lt1_clf_prob_sq'] = df['lt1_clf_prob_oof'] ** 2
        df['lt1_clf_prob_ge_0p8'] = (df['lt1_clf_prob_oof'] >= 0.8).astype(int)
        df['lt1_clf_prob_ge_0p9'] = (df['lt1_clf_prob_oof'] >= 0.9).astype(int)
        df['lt1_clf_prob_ge_0p95'] = (df['lt1_clf_prob_oof'] >= 0.95).astype(int)
    cols = ['lt1_clf_prob_oof', 'lt1_clf_prob_sq', 'lt1_clf_prob_ge_0p8', 'lt1_clf_prob_ge_0p9', 'lt1_clf_prob_ge_0p95']
    metrics = {'source': 'cached outputs/target_lt1_probability', 'note': 'Loaded OOF/test lt1 probabilities from previous target_lt1_features run.'}
    return (train_out, test_out, cols, metrics)

def fe_wide_tail_quantile_lgbm_build_feature_sets() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], dict]]]:
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(train, fe_wide_tail_quantile_lgbm_ROLLS, fe_wide_tail_quantile_lgbm_LAGS)
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_target_lt1, test_target_lt1, lt1_cols, lt1_metrics = fe_wide_tail_quantile_lgbm_load_or_make_lt1_features(train, test, x_base, xt_base, y, groups)
    target_lt1_feature_cols = list(dict.fromkeys(base_cols + lt1_cols))
    x_target_lt1, xt_target_lt1 = fe_neighbor_feature_missing_exps_fill_features(train_target_lt1, test_target_lt1, target_lt1_feature_cols, 'lag_roll_linear_interpolate')
    cache_dir = fe_wide_tail_quantile_lgbm_OUTPUT_DIR / 'feature_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    wt_train_path = cache_dir / 'wide_tail_train_probs.csv'
    wt_test_path = cache_dir / 'wide_tail_test_probs.csv'
    wt_meta_path = cache_dir / 'wide_tail_metrics.json'
    if wt_train_path.exists() and wt_test_path.exists() and wt_meta_path.exists():
        wt_train = pd.read_csv(wt_train_path)
        wt_test = pd.read_csv(wt_test_path)
        with open(wt_meta_path, 'r', encoding='utf-8') as f:
            wt_metrics = json.load(f)
        delay_cols = [c for c in wt_train.columns if c != 'ID']
        wt_train = wt_train.drop(columns=['ID'], errors='ignore')
        wt_test = wt_test.drop(columns=['ID'], errors='ignore')
    else:
        wt_train, wt_test, delay_cols, wt_metrics = fe_delay_bin_probability_features_make_delay_bin_prob_features(x=x_target_lt1, xt=xt_target_lt1, y=y, groups=groups, edges=fe_delay_bin_probability_features_BIN_SPECS['wide_tail'], spec_name='wide_tail')
        pd.concat([train[['ID']].reset_index(drop=True), wt_train.reset_index(drop=True)], axis=1).to_csv(wt_train_path, index=False)
        pd.concat([test[['ID']].reset_index(drop=True), wt_test.reset_index(drop=True)], axis=1).to_csv(wt_test_path, index=False)
        with open(wt_meta_path, 'w', encoding='utf-8') as f:
            json.dump(wt_metrics, f, ensure_ascii=False, indent=2, default=str)
    x_wide = pd.concat([x_target_lt1.reset_index(drop=True), wt_train.reset_index(drop=True)], axis=1)
    xt_wide = pd.concat([xt_target_lt1.reset_index(drop=True), wt_test.reset_index(drop=True)], axis=1)
    wide_cols = list(x_wide.columns)
    feature_sets = {'target_lt1_features': (x_target_lt1, xt_target_lt1, target_lt1_feature_cols, {'base_feature_set': 'target_lt1_probability_features_lgbm', 'base_public_lb': fe_wide_tail_quantile_lgbm_BASE_PUBLIC_LB, 'lt1_classifier_metrics': lt1_metrics}), 'wide_tail': (x_wide, xt_wide, wide_cols, {'base_feature_set': 'delay_bin_probability_wide_tail', 'base_public_lb_reference': fe_wide_tail_quantile_lgbm_BASE_PUBLIC_LB, 'lt1_classifier_metrics': lt1_metrics, 'delay_bin_classifier_metrics': wt_metrics, 'delay_bin_features': delay_cols})}
    return (train, test, y, groups, feature_sets)

def fe_wide_tail_quantile_lgbm_run_one(feature_set: str, alpha: float, x: pd.DataFrame, xt: pd.DataFrame, test: pd.DataFrame, y: pd.Series, groups: np.ndarray, feature_cols: list[str], metadata: dict) -> dict:
    tag = str(alpha).replace('.', 'p')
    name = f'quantile_{feature_set}_alpha_{tag}'
    output_dir = fe_wide_tail_quantile_lgbm_OUTPUT_DIR / name
    report_dir = output_dir / 'validation_reports'
    oof_dir = output_dir / 'oof_predictions'
    submission_dir = output_dir / 'submissions'
    importance_dir = output_dir / 'feature_importance'
    for path in [report_dir, oof_dir, submission_dir, importance_dir]:
        path.mkdir(parents=True, exist_ok=True)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(x), dtype=float)
    test_pred = np.zeros(len(xt), dtype=float)
    fold_rows = []
    importances = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = fe_wide_tail_quantile_lgbm_fit_quantile_lgbm(x.iloc[tr_idx], y.iloc[tr_idx], x.iloc[va_idx], y.iloc[va_idx], alpha)
        pred = fe_wide_tail_quantile_lgbm_predict_target(model, x.iloc[va_idx])
        oof[va_idx] = pred
        test_pred += fe_wide_tail_quantile_lgbm_predict_target(model, xt) / 5.0
        fold_rows.append({'model_name': name, 'validation': 'groupkfold', 'fold': fold, 'mae': float(mean_absolute_error(y.iloc[va_idx], pred)), 'best_iteration': int(model.best_iteration_ or fe_wide_tail_quantile_lgbm_make_params(alpha)['n_estimators'])})
        importances.append(pd.DataFrame({'model_name': name, 'feature': feature_cols, 'gain': model.booster_.feature_importance(importance_type='gain'), 'split': model.booster_.feature_importance(importance_type='split'), 'fold': fold}))
        print(f"{name} fold={fold} mae={fold_rows[-1]['mae']:.6f}", flush=True)
    group_oof = val_linear_make_prediction_frame(y.values, oof, groups=groups)
    group_summary = val_linear_summarize_prediction_frame(group_oof, 'groupkfold', name)
    group_bin = val_linear_make_bin_report(group_oof, 'groupkfold', name)
    tr_mask, va_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, val_linear_TargetHeavyConfig(random_state=42))
    exp_model = fe_wide_tail_quantile_lgbm_fit_quantile_lgbm(x.loc[tr_mask], y.loc[tr_mask], x.loc[va_mask], y.loc[va_mask], alpha)
    exp_train_pred = fe_wide_tail_quantile_lgbm_predict_target(exp_model, x.loc[tr_mask])
    exp_pred = fe_wide_tail_quantile_lgbm_predict_target(exp_model, x.loc[va_mask])
    exp_oof = val_linear_make_prediction_frame(y.loc[va_mask].values, exp_pred, groups=groups[va_mask])
    exp_summary = val_linear_summarize_prediction_frame(exp_oof, 'target_heavy_target_heavy_holdout', name)
    exp_summary['train_mae'] = float(mean_absolute_error(y.loc[tr_mask], exp_train_pred))
    exp_summary['train_val_gap'] = exp_summary['mae'] - exp_summary['train_mae']
    exp_bin = val_linear_make_bin_report(exp_oof, 'target_heavy_target_heavy_holdout', name)
    summary = pd.concat([group_summary, exp_summary], ignore_index=True)
    bin_report = pd.concat([group_bin, exp_bin], ignore_index=True)
    fold_report = pd.DataFrame(fold_rows)
    imp = pd.concat(importances, ignore_index=True)
    summary.to_csv(report_dir / f'{name}_summary.csv', index=False)
    bin_report.to_csv(report_dir / f'{name}_bin_report.csv', index=False)
    fold_report.to_csv(report_dir / f'{name}_fold_report.csv', index=False)
    scenario_stat.to_csv(report_dir / f'{name}_scenario_target_stat.csv', index=False)
    group_oof.to_csv(oof_dir / f'{name}_groupkfold_oof.csv', index=False)
    exp_oof.to_csv(oof_dir / f'{name}_target_heavy_target_heavy_holdout_oof.csv', index=False)
    imp.to_csv(importance_dir / f'{name}_feature_importance_by_fold.csv', index=False)
    imp.groupby('feature', as_index=False)[['gain', 'split']].mean().sort_values('gain', ascending=False).to_csv(importance_dir / f'{name}_feature_importance_mean.csv', index=False)
    sub = pd.read_csv('sample_submission.csv')[['ID']]
    pred_frame = pd.DataFrame({'ID': test['ID'].values, fe_neighbor_feature_missing_exps_TARGET: test_pred})
    sub = sub.merge(pred_frame, on='ID', how='left')
    sub[fe_neighbor_feature_missing_exps_TARGET] = sub[fe_neighbor_feature_missing_exps_TARGET].clip(lower=0)
    sub_path = submission_dir / f'{name}_submission.csv'
    sub.to_csv(sub_path, index=False)
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'hypothesis': 'LightGBM quantile regression to reduce high-delay underprediction.', 'feature_set': feature_set, 'alpha': alpha, 'objective': 'quantile', 'target_transform': 'none/raw target', 'feature_count': len(feature_cols), 'features': feature_cols, 'metadata': metadata}, f, ensure_ascii=False, indent=2, default=str)
    group_row = summary.loc[summary['validation'] == 'groupkfold'].iloc[0]
    exp_row = summary.loc[summary['validation'] == 'target_heavy_target_heavy_holdout'].iloc[0]
    return {'experiment': name, 'feature_set': feature_set, 'alpha': alpha, 'feature_count': len(feature_cols), 'groupkfold_mae': float(group_row['mae']), 'group_high50_mae': float(group_row['high50_mae']), 'group_high100_mae': float(group_row['high100_mae']), 'group_pred_mean': float(group_row['pred_mean']), 'group_pred_max': float(group_row['pred_max']), 'target_heavy_holdout_mae': float(exp_row['mae']), 'target_heavy_holdout_train_mae': float(exp_row['train_mae']), 'target_heavy_holdout_train_val_gap': float(exp_row['train_val_gap']), 'target_heavy_holdout_high50_mae': float(exp_row['high50_mae']), 'target_heavy_holdout_high100_mae': float(exp_row['high100_mae']), 'target_heavy_holdout_pred_mean': float(exp_row['pred_mean']), 'target_heavy_holdout_pred_max': float(exp_row['pred_max']), 'submission_path': str(sub_path)}

def fe_wide_tail_quantile_lgbm_main() -> None:
    fe_wide_tail_quantile_lgbm_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, y, groups, feature_sets = fe_wide_tail_quantile_lgbm_build_feature_sets()
    rows = []
    for feature_set in ['target_lt1_features', 'wide_tail']:
        x, xt, feature_cols, metadata = feature_sets[feature_set]
        for alpha in fe_wide_tail_quantile_lgbm_ALPHAS:
            result = fe_wide_tail_quantile_lgbm_run_one(feature_set, alpha, x, xt, test, y, groups, feature_cols, metadata)
            rows.append(result)
            pd.DataFrame(rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae']).to_csv(fe_wide_tail_quantile_lgbm_OUTPUT_DIR / 'quantile_target_lt1_widetail_summary.csv', index=False)
    with open(fe_wide_tail_quantile_lgbm_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'alphas': fe_wide_tail_quantile_lgbm_ALPHAS, 'feature_sets': list(feature_sets.keys()), 'base_public_lb': fe_wide_tail_quantile_lgbm_BASE_PUBLIC_LB, 'objective': 'LightGBM quantile on raw target'}, f, ensure_ascii=False, indent=2, default=str)
    out = pd.DataFrame(rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae'])
    print(out.to_string(index=False), flush=True)

# =============================================================================
# scenario rank 피처 생성
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
fe_scenario_rank_features_OUTPUT_DIR = Path('outputs/scenario_rank_features')
fe_scenario_rank_features_BASE_EXP = 'target_lt1_probability_features_lgbm'
fe_scenario_rank_features_ROLLS = (19, 20)
fe_scenario_rank_features_LAGS = (1, 2, 3, 5, 10)
fe_scenario_rank_features_RAW_RANK_COLS = ['order_inflow_15m', 'unique_sku_15m', 'urgent_order_ratio', 'robot_idle', 'robot_utilization', 'robot_charging', 'battery_mean', 'low_battery_ratio', 'charge_queue_length', 'avg_charge_wait', 'congestion_score', 'blocked_path_15m', 'near_collision_15m', 'pack_utilization', 'loading_dock_util', 'outbound_truck_wait_min']

def fe_scenario_rank_features_load_or_make_lt1_features(train, test, x_base, xt_base, y, groups):
    oof_path = Path('outputs/target_lt1_probability/oof_prob_target_lt1.csv')
    test_path = Path('outputs/target_lt1_probability/test_prob_target_lt1.csv')
    if not (oof_path.exists() and test_path.exists()):
        return fe_target_lt1_probability_make_low_target_prob_features(train, test, x_base, xt_base, y, groups)
    oof = pd.read_csv(oof_path)[['ID', 'lt1_clf_prob_oof']]
    te = pd.read_csv(test_path)[['ID', 'lt1_clf_prob']].rename(columns={'lt1_clf_prob': 'lt1_clf_prob_oof'})
    train_out = train.merge(oof, on='ID', how='left')
    test_out = test.merge(te, on='ID', how='left')
    for df in [train_out, test_out]:
        df['lt1_clf_prob_oof'] = df['lt1_clf_prob_oof'].fillna(0)
        df['lt1_clf_prob_sq'] = df['lt1_clf_prob_oof'] ** 2
        df['lt1_clf_prob_ge_0p8'] = (df['lt1_clf_prob_oof'] >= 0.8).astype(int)
        df['lt1_clf_prob_ge_0p9'] = (df['lt1_clf_prob_oof'] >= 0.9).astype(int)
        df['lt1_clf_prob_ge_0p95'] = (df['lt1_clf_prob_oof'] >= 0.95).astype(int)
    cols = ['lt1_clf_prob_oof', 'lt1_clf_prob_sq', 'lt1_clf_prob_ge_0p8', 'lt1_clf_prob_ge_0p9', 'lt1_clf_prob_ge_0p95']
    return (train_out, test_out, cols, {'source': 'cached target_lt1_features lt1 OOF/test probabilities'})

def fe_scenario_rank_features_add_scenario_raw_rank_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    created: list[str] = []
    cols = [c for c in fe_scenario_rank_features_RAW_RANK_COLS if c in train.columns and c in test.columns]
    for df in [train, test]:
        g = df.groupby('scenario_id', sort=False)
        for col in cols:
            s = df[col].astype(float)
            mean = g[col].transform('mean').astype(float)
            std = g[col].transform('std').astype(float).replace(0, np.nan)
            mx = g[col].transform('max').astype(float)
            mn = g[col].transform('min').astype(float)
            df[f'sc_rank_{col}'] = g[col].rank(pct=True, method='average').astype(float)
            df[f'sc_relmean_{col}'] = s - mean
            df[f'sc_z_{col}'] = ((s - mean) / (std + 1e-06)).fillna(0)
            df[f'sc_to_max_{col}'] = s / (mx.abs() + 1e-06)
            df[f'sc_range_pos_{col}'] = (s - mn) / ((mx - mn).abs() + 1e-06)
    for col in cols:
        created.extend([f'sc_rank_{col}', f'sc_relmean_{col}', f'sc_z_{col}', f'sc_to_max_{col}', f'sc_range_pos_{col}'])
    return created

def fe_scenario_rank_features_load_base_pred_context(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    oof_path = Path('outputs/target_lt1_probability_features_lgbm/oof_predictions/target_lt1_probability_features_lgbm_groupkfold_oof.csv')
    sub_path = Path('outputs/target_lt1_probability_features_lgbm/submissions/target_lt1_probability_features_lgbm_submission.csv')
    oof = pd.read_csv(oof_path)
    sub = pd.read_csv(sub_path)
    train = train.copy()
    test = test.copy()
    if len(oof) != len(train):
        raise ValueError('base OOF length mismatch')
    train['base_target_lt1_oof_pred'] = oof['pred'].values
    test = test.merge(sub.rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'base_target_lt1_oof_pred'}), on='ID', how='left')
    created = ['base_target_lt1_oof_pred']
    for df in [train, test]:
        g = df.groupby('scenario_id', sort=False)
        p = df['base_target_lt1_oof_pred'].astype(float)
        mean = g['base_target_lt1_oof_pred'].transform('mean').astype(float)
        std = g['base_target_lt1_oof_pred'].transform('std').astype(float).replace(0, np.nan)
        p90 = g['base_target_lt1_oof_pred'].transform(lambda s: s.quantile(0.9)).astype(float)
        p75 = g['base_target_lt1_oof_pred'].transform(lambda s: s.quantile(0.75)).astype(float)
        mx = g['base_target_lt1_oof_pred'].transform('max').astype(float)
        df['sc_pred_rank'] = g['base_target_lt1_oof_pred'].rank(pct=True, method='average').astype(float)
        df['sc_pred_relmean'] = p - mean
        df['sc_pred_z'] = ((p - mean) / (std + 1e-06)).fillna(0)
        df['sc_pred_to_p90'] = p / (p90.abs() + 1e-06)
        df['sc_pred_to_p75'] = p / (p75.abs() + 1e-06)
        df['sc_pred_to_max'] = p / (mx.abs() + 1e-06)
        df['sc_pred_mean'] = mean
        df['sc_pred_p75'] = p75
        df['sc_pred_p90'] = p90
        df['sc_pred_max'] = mx
        df['sc_pred_rank_x_pred'] = df['sc_pred_rank'] * p
    created.extend(['sc_pred_rank', 'sc_pred_relmean', 'sc_pred_z', 'sc_pred_to_p90', 'sc_pred_to_p75', 'sc_pred_to_max', 'sc_pred_mean', 'sc_pred_p75', 'sc_pred_p90', 'sc_pred_max', 'sc_pred_rank_x_pred'])
    return (train, test, created)

def fe_scenario_rank_features_main() -> None:
    fe_scenario_rank_features_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(train, fe_scenario_rank_features_ROLLS, fe_scenario_rank_features_LAGS)
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_target_lt1, test_target_lt1, lt1_cols, lt1_metrics = fe_scenario_rank_features_load_or_make_lt1_features(train, test, x_base, xt_base, y, groups)
    target_lt1_feature_cols = list(dict.fromkeys(base_cols + lt1_cols))
    train_raw = train_target_lt1.copy()
    test_raw = test_target_lt1.copy()
    raw_rank_cols = fe_scenario_rank_features_add_scenario_raw_rank_features(train_raw, test_raw)
    train_pred, test_pred, pred_rank_cols = fe_scenario_rank_features_load_base_pred_context(train_raw, test_raw)
    all_rank_cols = raw_rank_cols + pred_rank_cols
    experiments = [('scenario_raw_rank_features', raw_rank_cols, 'Add scenario-level raw feature rank, relative mean, z-score, max ratio, and range position features to target_lt1_features.'), ('scenario_raw_pred_rank_features', all_rank_cols, 'Add raw scenario rank features plus OOF base-prediction scenario context/rank features to target_lt1_features.')]
    rows = []
    for name, new_cols, hypothesis in experiments:
        result = fe_lgbm_log_target_exps_run_log_target_experiment(name=name, hypothesis=hypothesis, train=train_pred, test=test_pred, feature_cols=list(dict.fromkeys(target_lt1_feature_cols + new_cols)), metadata={**metadata, 'base_feature_set': fe_scenario_rank_features_BASE_EXP, 'rank_feature_count': len(new_cols), 'rank_features': new_cols, 'lt1_classifier_metrics': lt1_metrics, 'note': 'Scenario full-window aggregates use all rows within each scenario, target-free and available for static test rows.'})
        rows.append({'experiment': name, 'feature_count': len(target_lt1_feature_cols) + len(new_cols), 'new_feature_count': len(new_cols), **result, 'submission_path': str(Path('outputs') / name / 'submissions' / f'{name}_submission.csv')})
        pd.DataFrame(rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae']).to_csv(fe_scenario_rank_features_OUTPUT_DIR / 'scenario_rank_feature_summary.csv', index=False)
    with open(fe_scenario_rank_features_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_feature_set': fe_scenario_rank_features_BASE_EXP, 'roll_windows': fe_scenario_rank_features_ROLLS, 'lag_diff_steps': fe_scenario_rank_features_LAGS, 'raw_rank_cols': raw_rank_cols, 'pred_rank_cols': pred_rank_cols}, f, ensure_ascii=False, indent=2, default=str)
    print(pd.DataFrame(rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae']).to_string(index=False), flush=True)

# =============================================================================
# quantile blend와 layout_type 피처 실험
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
fe_quantile_blend_layout_type_OUTPUT_DIR = Path('outputs/quantile_blend_layout_type')
fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL = 'scenario_raw_pred_rank_features'
fe_quantile_blend_layout_type_WIDE_TAIL_QUANTILE = 'wide_tail_quantile_alpha_0p55'
fe_quantile_blend_layout_type_Q60 = 'wide_tail_quantile_alpha_0p60'
fe_quantile_blend_layout_type_WEIGHTS = [round(float(w), 3) for w in np.arange(0.0, 0.6001, 0.025)]

def fe_quantile_blend_layout_type_load_oof_scenario_rank(validation: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs') / fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL / 'oof_predictions' / f'{fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL}_{validation}_oof.csv')

def fe_quantile_blend_layout_type_load_oof_quantile(name: str, validation: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / name / 'oof_predictions' / f'{name}_{validation}_oof.csv')

def fe_quantile_blend_layout_type_load_sub_scenario_rank() -> pd.DataFrame:
    return pd.read_csv(Path('outputs') / fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL / 'submissions' / f'{fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL}_submission.csv')

def fe_quantile_blend_layout_type_load_sub_quantile(name: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / name / 'submissions' / f'{name}_submission.csv')

def fe_quantile_blend_layout_type_score_blend(q_name: str, weight: float, validation: str):
    b = fe_quantile_blend_layout_type_load_oof_scenario_rank(validation)
    q = fe_quantile_blend_layout_type_load_oof_quantile(q_name, validation)
    if len(b) != len(q) or not np.allclose(b['target'].values, q['target'].values):
        raise ValueError('OOF mismatch')
    pred = np.clip((1 - weight) * b['pred'].values + weight * q['pred'].values, 0, None)
    groups = b['group'].astype(str).values
    name = f"blend_{fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL}__{q_name}__w{str(weight).replace('.', 'p')}"
    frame = val_linear_make_prediction_frame(b['target'].values, pred, groups=groups)
    summ = val_linear_summarize_prediction_frame(frame, validation, name)
    binr = val_linear_make_bin_report(frame, validation, name)
    row = {'blend_name': name, 'quantile_model': q_name, 'quantile_weight': weight, 'validation': validation, 'mae': float(summ['mae'].iloc[0]), 'high50_mae': float(summ['high50_mae'].iloc[0]), 'high100_mae': float(summ['high100_mae'].iloc[0]), 'pred_mean': float(summ['pred_mean'].iloc[0]), 'pred_max': float(summ['pred_max'].iloc[0])}
    return (row, summ, binr)

def fe_quantile_blend_layout_type_make_blend_submission(q_name: str, weight: float, blend_name: str) -> str:
    out_dir = fe_quantile_blend_layout_type_OUTPUT_DIR / 'submissions'
    out_dir.mkdir(parents=True, exist_ok=True)
    b = fe_quantile_blend_layout_type_load_sub_scenario_rank().rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'base_pred'})
    q = fe_quantile_blend_layout_type_load_sub_quantile(q_name).rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'q_pred'})
    out = b.merge(q, on='ID', how='inner')
    out[fe_neighbor_feature_missing_exps_TARGET] = np.clip((1 - weight) * out['base_pred'].values + weight * out['q_pred'].values, 0, None)
    path = out_dir / f'{blend_name}_submission.csv'
    out[['ID', fe_neighbor_feature_missing_exps_TARGET]].to_csv(path, index=False)
    return str(path)

def fe_quantile_blend_layout_type_run_scenario_rank_blend_grid() -> pd.DataFrame:
    report_dir = fe_quantile_blend_layout_type_OUTPUT_DIR / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries, bins = ([], [], [])
    for q_name in [fe_quantile_blend_layout_type_WIDE_TAIL_QUANTILE, fe_quantile_blend_layout_type_Q60]:
        for w in fe_quantile_blend_layout_type_WEIGHTS:
            for val in ['groupkfold', 'target_heavy_target_heavy_holdout']:
                row, summ, binr = fe_quantile_blend_layout_type_score_blend(q_name, w, val)
                rows.append(row)
                summaries.append(summ)
                bins.append(binr)
    long = pd.DataFrame(rows)
    pivot_rows = []
    for (name, q_name, w), part in long.groupby(['blend_name', 'quantile_model', 'quantile_weight'], sort=False):
        g = part.loc[part.validation == 'groupkfold'].iloc[0]
        e = part.loc[part.validation == 'target_heavy_target_heavy_holdout'].iloc[0]
        pivot_rows.append({'blend_name': name, 'quantile_model': q_name, 'quantile_weight': float(w), 'groupkfold_mae': float(g.mae), 'group_high50_mae': float(g.high50_mae), 'group_high100_mae': float(g.high100_mae), 'group_pred_mean': float(g.pred_mean), 'group_pred_max': float(g.pred_max), 'target_heavy_holdout_mae': float(e.mae), 'target_heavy_holdout_high50_mae': float(e.high50_mae), 'target_heavy_holdout_high100_mae': float(e.high100_mae), 'target_heavy_holdout_pred_mean': float(e.pred_mean), 'target_heavy_holdout_pred_max': float(e.pred_max)})
    pivot = pd.DataFrame(pivot_rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae'])
    top = pivot.head(10).copy()
    top['submission_path'] = [fe_quantile_blend_layout_type_make_blend_submission(r.quantile_model, r.quantile_weight, r.blend_name) for r in top.itertuples(index=False)]
    long.to_csv(report_dir / 'scenario_rank_blend_long_metrics.csv', index=False)
    pivot.to_csv(report_dir / 'scenario_rank_blend_weight_grid_summary.csv', index=False)
    top.to_csv(report_dir / 'scenario_rank_blend_top10_groupkfold.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(report_dir / 'scenario_rank_blend_validation_summary.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(report_dir / 'scenario_rank_blend_bin_report.csv', index=False)
    return top

def fe_quantile_blend_layout_type_add_layout_type_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    layout = pd.read_csv('data/raw/layout_info.csv')[['layout_id', 'layout_type']]
    if 'layout_type' not in train.columns:
        train_type = train[['layout_id']].merge(layout, on='layout_id', how='left')['layout_type']
        train['layout_type'] = train_type.values
    if 'layout_type' not in test.columns:
        test_type = test[['layout_id']].merge(layout, on='layout_id', how='left')['layout_type']
        test['layout_type'] = test_type.values
    train['layout_type'] = train['layout_type'].astype(str).fillna('missing')
    test['layout_type'] = test['layout_type'].astype(str).fillna('missing')
    freq = train['layout_type'].value_counts(normalize=True)
    train['layout_type_freq'] = train['layout_type'].map(freq).fillna(0)
    test['layout_type_freq'] = test['layout_type'].map(freq).fillna(0)
    cats = sorted(set(train['layout_type']) | set(test['layout_type']))
    created = ['layout_type_freq']
    for cat in cats:
        safe = ''.join((ch if ch.isalnum() else '_' for ch in cat.lower()))
        col = f'layout_type_oh_{safe}'
        train[col] = (train['layout_type'] == cat).astype(np.int8)
        test[col] = (test['layout_type'] == cat).astype(np.int8)
        created.append(col)
    return created

def fe_quantile_blend_layout_type_run_layout_type_experiment() -> pd.DataFrame:
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(train, fe_scenario_rank_features_ROLLS, fe_scenario_rank_features_LAGS)
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_target_lt1, test_target_lt1, lt1_cols, lt1_metrics = fe_scenario_rank_features_load_or_make_lt1_features(train, test, x_base, xt_base, y, groups)
    target_lt1_feature_cols = list(dict.fromkeys(base_cols + lt1_cols))
    raw_rank_cols = fe_scenario_rank_features_add_scenario_raw_rank_features(train_target_lt1, test_target_lt1)
    train_pred, test_pred, pred_rank_cols = fe_scenario_rank_features_load_base_pred_context(train_target_lt1, test_target_lt1)
    layout_cols = fe_quantile_blend_layout_type_add_layout_type_features(train_pred, test_pred)
    feature_cols = list(dict.fromkeys(target_lt1_feature_cols + raw_rank_cols + pred_rank_cols + layout_cols))
    name = 'layout_type_scenario_rank_features'
    result = fe_lgbm_log_target_exps_run_log_target_experiment(name=name, hypothesis='Add layout_type one-hot and frequency encoding to scenario raw-pred rank feature set.', train=train_pred, test=test_pred, feature_cols=feature_cols, metadata={**metadata, 'base_feature_set': fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL, 'layout_type_features': layout_cols, 'lt1_classifier_metrics': lt1_metrics})
    row = pd.DataFrame([{'experiment': name, 'feature_count': len(feature_cols), 'new_layout_type_feature_count': len(layout_cols), **result, 'submission_path': str(Path('outputs') / name / 'submissions' / f'{name}_submission.csv')}])
    row.to_csv(fe_quantile_blend_layout_type_OUTPUT_DIR / 'reports' / 'layout_type_summary.csv', index=False)
    return row

def fe_quantile_blend_layout_type_main() -> None:
    fe_quantile_blend_layout_type_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_blends = fe_quantile_blend_layout_type_run_scenario_rank_blend_grid()
    layout_result = fe_quantile_blend_layout_type_run_layout_type_experiment()
    with open(fe_quantile_blend_layout_type_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_model': fe_quantile_blend_layout_type_SCENARIO_RANK_MODEL, 'quantile_models': [fe_quantile_blend_layout_type_WIDE_TAIL_QUANTILE, fe_quantile_blend_layout_type_Q60], 'weights': fe_quantile_blend_layout_type_WEIGHTS, 'layout_type_experiment': 'layout_type_scenario_rank_features'}, f, ensure_ascii=False, indent=2, default=str)
    print('Top scenario-rank blends')
    print(top_blends.to_string(index=False), flush=True)
    print('\nLayout type')
    print(layout_result.to_string(index=False), flush=True)

# =============================================================================
# context rolling 피처와 최종 feature set
# =============================================================================

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
fe_context_rolling_features_OUTPUT_DIR = Path('outputs/context_rolling_features')
fe_context_rolling_features_BASE_NAME = 'context_rolling_lgbm'
fe_context_rolling_features_WIDE_TAIL_QUANTILE = 'wide_tail_quantile_alpha_0p55'
fe_context_rolling_features_Q60 = 'wide_tail_quantile_alpha_0p60'
fe_context_rolling_features_WEIGHTS = [round(float(w), 3) for w in np.arange(0.0, 0.6001, 0.025)]
fe_context_rolling_features_EPS = 1e-06
fe_context_rolling_features_CONTEXT_ROLLS = [19, 20, 25]
warnings.simplefilter('ignore', PerformanceWarning)

def fe_context_rolling_features_safe_col(df: pd.DataFrame, col: str, default: float=0.0) -> pd.Series:
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)

def fe_context_rolling_features_add_backlog_diff_cumulative_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    created: list[str] = []
    for df in [train, test]:
        order = fe_context_rolling_features_safe_col(df, 'order_inflow_15m')
        pack_station = fe_context_rolling_features_safe_col(df, 'pack_station_count')
        robot_active = fe_context_rolling_features_safe_col(df, 'robot_active')
        robot_total = fe_context_rolling_features_safe_col(df, 'robot_total')
        pack_util = fe_context_rolling_features_safe_col(df, 'pack_utilization')
        robot_util = fe_context_rolling_features_safe_col(df, 'robot_utilization')
        congestion = fe_context_rolling_features_safe_col(df, 'congestion_score')
        low_batt = fe_context_rolling_features_safe_col(df, 'low_battery_ratio')
        charge_q = fe_context_rolling_features_safe_col(df, 'charge_queue_length')
        backorder = fe_context_rolling_features_safe_col(df, 'backorder_ratio')
        loading = fe_context_rolling_features_safe_col(df, 'loading_dock_util')
        label_q = fe_context_rolling_features_safe_col(df, 'label_print_queue')
        truck_wait = fe_context_rolling_features_safe_col(df, 'outbound_truck_wait_min')
        df['bp_order_pack_capacity_gap'] = order - pack_station * (1.0 - pack_util)
        df['bp_order_robot_capacity_gap'] = order - robot_active * (1.0 - robot_util)
        df['bp_robot_available_gap'] = robot_total - robot_active - fe_context_rolling_features_safe_col(df, 'robot_charging')
        df['bp_pack_overload'] = np.maximum(df['bp_order_pack_capacity_gap'], 0)
        df['bp_robot_overload'] = np.maximum(df['bp_order_robot_capacity_gap'], 0)
        df['bp_demand_capacity_pressure'] = order / (pack_station + robot_active + fe_context_rolling_features_EPS)
        df['bp_downstream_pressure'] = (pack_util + loading + label_q / (label_q.abs().median() + fe_context_rolling_features_EPS) + truck_wait / (truck_wait.abs().median() + fe_context_rolling_features_EPS)) / 4.0
        df['bp_congestion_order_pressure'] = congestion * df['bp_demand_capacity_pressure']
        df['bp_congestion_pack_overload'] = congestion * df['bp_pack_overload']
        df['bp_battery_robot_overload'] = (low_batt + charge_q / (charge_q.abs().median() + fe_context_rolling_features_EPS)) * df['bp_robot_overload']
        df['bp_backorder_pack_pressure'] = backorder * df['bp_pack_overload']
        g = df.groupby('scenario_id', sort=False)
        for col in ['bp_pack_overload', 'bp_robot_overload', 'bp_demand_capacity_pressure', 'bp_congestion_order_pressure', 'bp_downstream_pressure']:
            shifted = df[col].groupby(df['scenario_id'], sort=False).shift(1).fillna(0)
            df[f'{col}_cumsum_prev'] = shifted.groupby(df['scenario_id'], sort=False).cumsum()
            df[f'{col}_cummean_prev'] = shifted.groupby(df['scenario_id'], sort=False).expanding().mean().reset_index(level=0, drop=True).fillna(0).values
            diff = df[col] - df[col].groupby(df['scenario_id'], sort=False).shift(1)
            df[f'{col}_diff1'] = diff.fillna(0)
            df[f'{col}_diff_x_cumsum'] = df[f'{col}_diff1'] * df[f'{col}_cumsum_prev']
            df[f'{col}_diff_x_cummean'] = df[f'{col}_diff1'] * df[f'{col}_cummean_prev']
        for raw_col, accum_col in [('congestion_score', 'bp_congestion_order_pressure_cumsum_prev'), ('pack_utilization', 'bp_pack_overload_cumsum_prev'), ('order_inflow_15m', 'bp_demand_capacity_pressure_cumsum_prev'), ('low_battery_ratio', 'bp_robot_overload_cumsum_prev')]:
            if raw_col in df.columns and accum_col in df.columns:
                d = df[raw_col].astype(float) - df[raw_col].astype(float).groupby(df['scenario_id'], sort=False).shift(1)
                df[f'bp_{raw_col}_diff1_x_{accum_col}'] = d.fillna(0) * df[accum_col]
    base = ['bp_order_pack_capacity_gap', 'bp_order_robot_capacity_gap', 'bp_robot_available_gap', 'bp_pack_overload', 'bp_robot_overload', 'bp_demand_capacity_pressure', 'bp_downstream_pressure', 'bp_congestion_order_pressure', 'bp_congestion_pack_overload', 'bp_battery_robot_overload', 'bp_backorder_pack_pressure']
    created.extend(base)
    for col in ['bp_pack_overload', 'bp_robot_overload', 'bp_demand_capacity_pressure', 'bp_congestion_order_pressure', 'bp_downstream_pressure']:
        created.extend([f'{col}_cumsum_prev', f'{col}_cummean_prev', f'{col}_diff1', f'{col}_diff_x_cumsum', f'{col}_diff_x_cummean'])
    for raw_col, accum_col in [('congestion_score', 'bp_congestion_order_pressure_cumsum_prev'), ('pack_utilization', 'bp_pack_overload_cumsum_prev'), ('order_inflow_15m', 'bp_demand_capacity_pressure_cumsum_prev'), ('low_battery_ratio', 'bp_robot_overload_cumsum_prev')]:
        created.append(f'bp_{raw_col}_diff1_x_{accum_col}')
    return [c for c in dict.fromkeys(created) if c in train.columns]

def fe_context_rolling_features_add_context_roll_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """동적으로 변하는 context 피처만 rolling합니다. scenario/layout 상수 피처는 제외합니다."""
    created: list[str] = []
    candidates = ['base_target_lt1_oof_pred', 'sc_pred_rank', 'sc_pred_relmean', 'sc_pred_z', 'sc_pred_to_p90', 'sc_pred_to_p75', 'sc_pred_to_max', 'sc_pred_rank_x_pred', 'bp_pack_overload', 'bp_robot_overload', 'bp_demand_capacity_pressure', 'bp_downstream_pressure', 'bp_congestion_order_pressure', 'bp_congestion_pack_overload', 'bp_battery_robot_overload', 'bp_backorder_pack_pressure', 'bp_order_pack_capacity_gap', 'bp_order_robot_capacity_gap', 'bp_robot_available_gap']
    candidates.extend([c for c in train.columns if c.startswith('sc_rank_') or c.startswith('sc_z_') or c.startswith('sc_relmean_') or c.startswith('sc_to_max_') or c.startswith('sc_range_pos_')])
    candidates = [c for c in dict.fromkeys(candidates) if c in train.columns and c in test.columns]
    for df in [train, test]:
        gkey = df['scenario_id']
        new_cols: dict[str, np.ndarray] = {}
        for col in candidates:
            s = df[col].astype(float)
            prev = s.groupby(gkey, sort=False).shift(1)
            for w in fe_context_rolling_features_CONTEXT_ROLLS:
                roll = prev.groupby(gkey, sort=False).rolling(w, min_periods=1)
                mean = roll.mean().reset_index(level=0, drop=True)
                mx = roll.max().reset_index(level=0, drop=True)
                std = roll.std().reset_index(level=0, drop=True)
                new_cols[f'{col}_ctxroll{w}_mean'] = mean.fillna(0).to_numpy()
                new_cols[f'{col}_ctxroll{w}_max'] = mx.fillna(0).to_numpy()
                new_cols[f'{col}_ctxroll{w}_std'] = std.fillna(0).to_numpy()
                new_cols[f'{col}_vs_ctxroll{w}_mean'] = (s - mean).fillna(0).to_numpy()
        if new_cols:
            df[list(new_cols)] = pd.DataFrame(new_cols, index=df.index)
    for col in candidates:
        for w in fe_context_rolling_features_CONTEXT_ROLLS:
            created.extend([f'{col}_ctxroll{w}_mean', f'{col}_ctxroll{w}_max', f'{col}_ctxroll{w}_std', f'{col}_vs_ctxroll{w}_mean'])
    return [c for c in dict.fromkeys(created) if c in train.columns]

def fe_context_rolling_features_build_feature_set():
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    bp_cols = fe_context_rolling_features_add_backlog_diff_cumulative_features(train, test)
    base_cols = fe_log_roll_window_search_select_columns(train, fe_scenario_rank_features_ROLLS, fe_scenario_rank_features_LAGS)
    base_cols = list(dict.fromkeys(base_cols + bp_cols))
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    train_target_lt1, test_target_lt1, lt1_cols, lt1_metrics = fe_scenario_rank_features_load_or_make_lt1_features(train, test, x_base, xt_base, y, groups)
    target_lt1_feature_cols = list(dict.fromkeys(base_cols + lt1_cols))
    raw_rank_cols = fe_scenario_rank_features_add_scenario_raw_rank_features(train_target_lt1, test_target_lt1)
    train_pred, test_pred, pred_rank_cols = fe_scenario_rank_features_load_base_pred_context(train_target_lt1, test_target_lt1)
    layout_cols = fe_quantile_blend_layout_type_add_layout_type_features(train_pred, test_pred)
    context_roll_cols = fe_context_rolling_features_add_context_roll_features(train_pred, test_pred)
    feature_cols = list(dict.fromkeys(target_lt1_feature_cols + raw_rank_cols + pred_rank_cols + layout_cols + context_roll_cols))
    metadata = {**metadata, 'base_feature_set': 'layout_type_scenario_rank_features', 'backlog_diff_cumulative_features': bp_cols, 'context_roll_features': context_roll_cols, 'context_roll_windows': fe_context_rolling_features_CONTEXT_ROLLS, 'context_roll_policy': 'shift(1) within scenario; dynamic prediction/rank/backlog context only; duplicate column names removed', 'layout_type_features': layout_cols, 'lt1_classifier_metrics': lt1_metrics, 'duplicate_policy': 'features already present in selected columns are deduplicated by dict.fromkeys'}
    return (train_pred, test_pred, feature_cols, metadata)

def fe_context_rolling_features_load_base_oof(validation: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs') / fe_context_rolling_features_BASE_NAME / 'oof_predictions' / f'{fe_context_rolling_features_BASE_NAME}_{validation}_oof.csv')

def fe_context_rolling_features_load_q_oof(q: str, validation: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / q / 'oof_predictions' / f'{q}_{validation}_oof.csv')

def fe_context_rolling_features_load_base_sub() -> pd.DataFrame:
    return pd.read_csv(Path('outputs') / fe_context_rolling_features_BASE_NAME / 'submissions' / f'{fe_context_rolling_features_BASE_NAME}_submission.csv')

def fe_context_rolling_features_load_q_sub(q: str) -> pd.DataFrame:
    return pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / q / 'submissions' / f'{q}_submission.csv')

def fe_context_rolling_features_blend_grid() -> pd.DataFrame:
    report_dir = fe_context_rolling_features_OUTPUT_DIR / 'reports'
    sub_dir = fe_context_rolling_features_OUTPUT_DIR / 'submissions'
    report_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries, bins = ([], [], [])
    for q in [fe_context_rolling_features_WIDE_TAIL_QUANTILE, fe_context_rolling_features_Q60]:
        for w in fe_context_rolling_features_WEIGHTS:
            for val in ['groupkfold', 'target_heavy_target_heavy_holdout']:
                b = fe_context_rolling_features_load_base_oof(val)
                qq = fe_context_rolling_features_load_q_oof(q, val)
                pred = np.clip((1 - w) * b['pred'].values + w * qq['pred'].values, 0, None)
                name = f"blend_{fe_context_rolling_features_BASE_NAME}__{q}__w{str(w).replace('.', 'p')}"
                frame = val_linear_make_prediction_frame(b['target'].values, pred, groups=b['group'].astype(str).values)
                summ = val_linear_summarize_prediction_frame(frame, val, name)
                binr = val_linear_make_bin_report(frame, val, name)
                rows.append({'blend_name': name, 'quantile_model': q, 'quantile_weight': w, 'validation': val, 'mae': float(summ.mae.iloc[0]), 'high50_mae': float(summ.high50_mae.iloc[0]), 'high100_mae': float(summ.high100_mae.iloc[0]), 'pred_mean': float(summ.pred_mean.iloc[0]), 'pred_max': float(summ.pred_max.iloc[0])})
                summaries.append(summ)
                bins.append(binr)
    long = pd.DataFrame(rows)
    pivot_rows = []
    for (name, q, w), part in long.groupby(['blend_name', 'quantile_model', 'quantile_weight'], sort=False):
        g = part.loc[part.validation == 'groupkfold'].iloc[0]
        e = part.loc[part.validation == 'target_heavy_target_heavy_holdout'].iloc[0]
        pivot_rows.append({'blend_name': name, 'quantile_model': q, 'quantile_weight': float(w), 'groupkfold_mae': float(g.mae), 'group_high50_mae': float(g.high50_mae), 'group_high100_mae': float(g.high100_mae), 'group_pred_mean': float(g.pred_mean), 'group_pred_max': float(g.pred_max), 'target_heavy_holdout_mae': float(e.mae), 'target_heavy_holdout_high50_mae': float(e.high50_mae), 'target_heavy_holdout_high100_mae': float(e.high100_mae), 'target_heavy_holdout_pred_mean': float(e.pred_mean), 'target_heavy_holdout_pred_max': float(e.pred_max)})
    pivot = pd.DataFrame(pivot_rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae'])
    top = pivot.head(12).copy()
    base_sub = fe_context_rolling_features_load_base_sub().rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'base_pred'})
    paths = []
    for r in top.itertuples(index=False):
        qsub = fe_context_rolling_features_load_q_sub(r.quantile_model).rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'q_pred'})
        out = base_sub.merge(qsub, on='ID', how='inner')
        out[fe_neighbor_feature_missing_exps_TARGET] = np.clip((1 - r.quantile_weight) * out['base_pred'].values + r.quantile_weight * out['q_pred'].values, 0, None)
        path = sub_dir / f'{r.blend_name}_submission.csv'
        out[['ID', fe_neighbor_feature_missing_exps_TARGET]].to_csv(path, index=False)
        paths.append(str(path))
    top['submission_path'] = paths
    long.to_csv(report_dir / 'backlog_blend_long_metrics.csv', index=False)
    pivot.to_csv(report_dir / 'backlog_blend_weight_grid_summary.csv', index=False)
    top.to_csv(report_dir / 'backlog_blend_top12_groupkfold.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(report_dir / 'backlog_blend_validation_summary.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(report_dir / 'backlog_blend_bin_report.csv', index=False)
    return top

def fe_context_rolling_features_main() -> None:
    fe_context_rolling_features_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, feature_cols, metadata = fe_context_rolling_features_build_feature_set()
    (fe_context_rolling_features_OUTPUT_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    result = fe_lgbm_log_target_exps_run_log_target_experiment(name=fe_context_rolling_features_BASE_NAME, hypothesis='Add non-duplicate backlog proxy, cumulative backlog, and diff x cumulative pressure features to the layout/scenario-rank base model.', train=train, test=test, feature_cols=feature_cols, metadata=metadata)
    pd.DataFrame([{'experiment': fe_context_rolling_features_BASE_NAME, 'feature_count': len(feature_cols), **result}]).to_csv(fe_context_rolling_features_OUTPUT_DIR / 'reports' / 'context_roll_base_summary.csv', index=False)
    top = fe_context_rolling_features_blend_grid()
    with open(fe_context_rolling_features_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_model': fe_context_rolling_features_BASE_NAME, 'feature_count': len(feature_cols), 'quantile_models': [fe_context_rolling_features_WIDE_TAIL_QUANTILE, fe_context_rolling_features_Q60], 'weights': fe_context_rolling_features_WEIGHTS, 'metadata': metadata}, f, ensure_ascii=False, indent=2, default=str)
    print(pd.DataFrame([{'experiment': fe_context_rolling_features_BASE_NAME, 'feature_count': len(feature_cols), **result}]).to_string(index=False), flush=True)
    print(top.to_string(index=False), flush=True)


@dataclass
class BaseFeatureData:
    train: pd.DataFrame
    test: pd.DataFrame
    x_base: pd.DataFrame
    xt_base: pd.DataFrame
    y: pd.Series
    groups: np.ndarray
    base_cols: list[str]
    metadata: dict


def setup_feature_runtime() -> None:
    prepare_runtime(SRC_DIR)


def build_base_lag_roll_features() -> BaseFeatureData:
    setup_feature_runtime()
    train, test, metadata = fe_log_roll_window_search_build_search_store()
    base_cols = fe_log_roll_window_search_select_columns(
        train,
        fe_target_lt1_probability_ROLLS,
        fe_target_lt1_probability_LAGS,
    )
    x_base, xt_base = fe_neighbor_feature_missing_exps_fill_features(train, test, base_cols, "lag_roll_linear_interpolate")
    y = train[TARGET].astype(float)
    groups = train["scenario_id"].values
    return BaseFeatureData(train, test, x_base, xt_base, y, groups, base_cols, metadata)


def add_target_lt1_probability_features(base_data: BaseFeatureData) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    setup_feature_runtime()
    return fe_target_lt1_probability_make_low_target_prob_features(
        train=base_data.train,
        test=base_data.test,
        x=base_data.x_base,
        xt=base_data.xt_base,
        y=base_data.y,
        groups=base_data.groups,
    )


def build_low_delay_probability_outputs() -> None:
    setup_feature_runtime()
    fe_target_lt1_probability_main()


def build_wide_tail_quantile_features():
    setup_feature_runtime()
    return fe_wide_tail_quantile_lgbm_build_feature_sets()


def add_backlog_diff_cumulative_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    setup_feature_runtime()
    return fe_context_rolling_features_add_backlog_diff_cumulative_features(train, test)


def add_context_rolling_features(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    setup_feature_runtime()
    return fe_context_rolling_features_add_context_roll_features(train, test)


def build_context_rolling_feature_set() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    setup_feature_runtime()
    return fe_context_rolling_features_build_feature_set()


def describe_feature_outputs() -> pd.DataFrame:
    rows = [
        {"stage": "target_lt1_probability", "path": OUTPUTS_DIR / "target_lt1_probability" / "oof_prob_target_lt1.csv"},
        {"stage": "wide_tail_probability", "path": OUTPUTS_DIR / "wide_tail_quantile_lgbm" / "feature_cache" / "wide_tail_train_probs.csv"},
        {
            "stage": "context_rolling_groupkfold_oof",
            "path": OUTPUTS_DIR / "context_rolling_lgbm" / "oof_predictions" / "context_rolling_lgbm_groupkfold_oof.csv",
        },
    ]
    out = pd.DataFrame(rows)
    out["exists"] = out["path"].map(lambda path: Path(path).exists())
    return out


def main() -> None:
    print(describe_feature_outputs().to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
