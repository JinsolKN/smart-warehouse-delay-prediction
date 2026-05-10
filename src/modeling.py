from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CONTEXT_ROLLING_MODEL_NAME,
    LOW_DELAY_POSTPROCESS_SUBMISSION_NAME,
    FINAL_RETRAINED_PATH,
    OUTPUTS_DIR,
    WIDE_TAIL_QUANTILE_MODEL_NAME,
    RESULTS_SUBMISSIONS_DIR,
    RESULTS_VALIDATION_DIR,
    ROOT,
    SRC_DIR,
    TARGET,
    ensure_result_dirs,
)
from data_io import prepare_runtime
from feature_engineering import *
from validation import compare_submission_files, validate_submission_file


# =============================================================================
# 고지연/저지연 specialist blend
# =============================================================================

import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
SPECIALIST_BLEND_OUTPUT_DIR = Path('outputs/high_low_specialist_blend')
SPECIALIST_BLEND_NAME = 'high_low_specialist_blend'
SPECIALIST_BASE_MODEL_NAME = 'context_rolling_lgbm'
QUANTILE_BLEND_WEIGHT = 0.325
SPECIALIST_EARLY_STOPPING_ROUNDS = 50
SPECIALIST_RANDOM_STATE = 42

def fit_specialist_lgbm(x_tr, y_tr_log, x_va, y_va_log, sample_weight):
    params = {**fe_experiments_LGBM_PARAMS, 'objective': 'mae', 'metric': 'mae', 'random_state': SPECIALIST_RANDOM_STATE}
    model = lgb.LGBMRegressor(**params)
    model.fit(x_tr, y_tr_log, sample_weight=sample_weight, eval_set=[(x_va, y_va_log)], callbacks=[lgb.early_stopping(SPECIALIST_EARLY_STOPPING_ROUNDS, verbose=False)])
    return model

def predict_log_target_lgbm(model, x):
    return np.clip(np.expm1(model.predict(x)), 0, None)

def make_high_delay_weights(y: pd.Series) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    w[y.values >= 40] = 2.0
    w[y.values >= 50] = 4.0
    w[y.values >= 100] = 8.0
    return w

def make_low_delay_weights(y: pd.Series) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    w[y.values < 20] = 2.0
    w[y.values < 10] = 4.0
    w[y.values < 1] = 8.0
    return w

def load_context_quantile_blend_oof(validation: str) -> pd.DataFrame:
    base_oof = pd.read_csv(Path('outputs') / SPECIALIST_BASE_MODEL_NAME / 'oof_predictions' / f'{SPECIALIST_BASE_MODEL_NAME}_{validation}_oof.csv')
    q_oof = pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / fe_context_rolling_features_WIDE_TAIL_QUANTILE / 'oof_predictions' / f'{fe_context_rolling_features_WIDE_TAIL_QUANTILE}_{validation}_oof.csv')
    out = base_oof.copy()
    out['pred'] = np.clip((1 - QUANTILE_BLEND_WEIGHT) * base_oof['pred'].values + QUANTILE_BLEND_WEIGHT * q_oof['pred'].values, 0, None)
    return out

def load_context_quantile_blend_submission(test_ids: pd.Series) -> pd.DataFrame:
    base_sub = pd.read_csv(Path('outputs') / SPECIALIST_BASE_MODEL_NAME / 'submissions' / f'{SPECIALIST_BASE_MODEL_NAME}_submission.csv').rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'base_raw_pred'})
    q_sub = pd.read_csv(Path('outputs/wide_tail_quantile_lgbm') / fe_context_rolling_features_WIDE_TAIL_QUANTILE / 'submissions' / f'{fe_context_rolling_features_WIDE_TAIL_QUANTILE}_submission.csv').rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'q_pred'})
    merged = pd.DataFrame({'ID': test_ids.values}).merge(base_sub, on='ID', how='left').merge(q_sub, on='ID', how='left')
    merged['pred'] = np.clip((1 - QUANTILE_BLEND_WEIGHT) * merged['base_raw_pred'].values + QUANTILE_BLEND_WEIGHT * merged['q_pred'].values, 0, None)
    return merged[['ID', 'pred']]

def make_delay_gate_probabilities(train, test, x, xt, y, groups):
    existing = SPECIALIST_BLEND_OUTPUT_DIR.parent / 'delay_gate_probabilities' / 'oof_predictions'
    g_path = existing / 'delay_gate_probability_lgbm_groupkfold_gate_probs.csv'
    e_path = existing / 'delay_gate_probability_lgbm_target_heavy_target_heavy_holdout_gate_probs.csv'
    t_path = existing / 'delay_gate_probability_lgbm_test_gate_probs.csv'
    if g_path.exists() and e_path.exists() and t_path.exists():
        return (pd.read_csv(g_path), pd.read_csv(e_path), pd.read_csv(t_path))
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 900, 'learning_rate': 0.035, 'num_leaves': 63, 'min_child_samples': 30, 'subsample': 0.85, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 2.0, 'random_state': SPECIALIST_RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
    labels = {'lt1_prob': (y < 1).astype(int), 'high50_prob': (y >= 50).astype(int), 'high100_prob': (y >= 100).astype(int)}
    group_probs = pd.DataFrame(index=train.index)
    test_probs = pd.DataFrame(index=test.index)
    gkf = GroupKFold(n_splits=5)
    for name, label in labels.items():
        oof = np.zeros(len(x))
        tst = np.zeros(len(xt))
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
            clf = lgb.LGBMClassifier(**{**params, 'random_state': SPECIALIST_RANDOM_STATE + fold})
            clf.fit(x.iloc[tr_idx], label.iloc[tr_idx], eval_set=[(x.iloc[va_idx], label.iloc[va_idx])], callbacks=[lgb.early_stopping(15, verbose=False)])
            oof[va_idx] = clf.predict_proba(x.iloc[va_idx])[:, 1]
            tst += clf.predict_proba(xt)[:, 1] / 5
        group_probs[name] = oof
        test_probs[name] = tst
    tr_mask, va_mask, _ = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, val_linear_TargetHeavyConfig(random_state=42))
    exp_probs = pd.DataFrame(index=np.arange(va_mask.sum()))
    for name, label in labels.items():
        clf = lgb.LGBMClassifier(**params)
        clf.fit(x.loc[tr_mask], label.loc[tr_mask], eval_set=[(x.loc[va_mask], label.loc[va_mask])], callbacks=[lgb.early_stopping(15, verbose=False)])
        exp_probs[name] = clf.predict_proba(x.loc[va_mask])[:, 1]
    existing.mkdir(parents=True, exist_ok=True)
    group_probs.to_csv(g_path, index=False)
    exp_probs.to_csv(e_path, index=False)
    test_probs.to_csv(t_path, index=False)
    return (group_probs, exp_probs, test_probs)

def train_weighted_specialist_model(name, x, xt, y, groups, weights):
    y_log = np.log1p(y)
    oof = np.zeros(len(x))
    test_pred = np.zeros(len(xt))
    rows = []
    gkf = GroupKFold(n_splits=5)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(x, y, groups), start=1):
        model = fit_specialist_lgbm(x.iloc[tr_idx], y_log.iloc[tr_idx], x.iloc[va_idx], y_log.iloc[va_idx], weights[tr_idx])
        pred = predict_log_target_lgbm(model, x.iloc[va_idx])
        oof[va_idx] = pred
        test_pred += predict_log_target_lgbm(model, xt) / 5
        rows.append({'model_name': name, 'fold': fold, 'mae': float(mean_absolute_error(y.iloc[va_idx], pred)), 'best_iteration': int(model.best_iteration_ or fe_experiments_LGBM_PARAMS['n_estimators'])})
    return (oof, test_pred, pd.DataFrame(rows))

def train_specialist_target_heavy_holdout(name, x, y, groups, weights):
    tr_mask, va_mask, scenario_stat = val_linear_make_target_heavy_target_heavy_holdout_split(y.values, groups, val_linear_TargetHeavyConfig(random_state=42))
    model = fit_specialist_lgbm(x.loc[tr_mask], np.log1p(y.loc[tr_mask]), x.loc[va_mask], np.log1p(y.loc[va_mask]), weights[tr_mask])
    pred = predict_log_target_lgbm(model, x.loc[va_mask])
    return (pred, tr_mask, va_mask, scenario_stat)

def summarize_specialist_predictions(name, validation, y_true, pred, groups):
    frame = val_linear_make_prediction_frame(y_true, np.clip(pred, 0, None), groups=groups)
    return (val_linear_summarize_prediction_frame(frame, validation, name), val_linear_make_bin_report(frame, validation, name))

def search_specialist_blend_weights(group_base, exp_base, base_sub, high_group, low_group, high_exp, low_exp, high_test, low_test, group_probs, exp_probs, test_probs, exp_groups, test_ids):
    report_dir = SPECIALIST_BLEND_OUTPUT_DIR / 'reports'
    sub_dir = SPECIALIST_BLEND_OUTPUT_DIR / 'submissions'
    rows, summaries, bins = ([], [], [])
    high_base_grid = [0.0, 0.025, 0.05, 0.075]
    high50_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    high100_grid = [0.0, 0.1, 0.2, 0.4, 0.6]
    max_high_grid = [0.15, 0.25, 0.35, 0.45]
    low_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    max_low_grid = [0.1, 0.2, 0.3]
    for hb in high_base_grid:
        for h50 in high50_grid:
            for h100 in high100_grid:
                for mh in max_high_grid:
                    ghw = np.clip(hb + h50 * group_probs['high50_prob'].values + h100 * group_probs['high100_prob'].values, 0, mh)
                    ehw = np.clip(hb + h50 * exp_probs['high50_prob'].values + h100 * exp_probs['high100_prob'].values, 0, mh)
                    for lc in low_grid:
                        for ml in max_low_grid:
                            glw = np.clip(lc * group_probs['lt1_prob'].values, 0, ml)
                            elw = np.clip(lc * exp_probs['lt1_prob'].values, 0, ml)
                            gp = np.clip(group_base['pred'].values + ghw * (high_group - group_base['pred'].values) + glw * (low_group - group_base['pred'].values), 0, None)
                            ep = np.clip(exp_base['pred'].values + ehw * (high_exp - exp_base['pred'].values) + elw * (low_exp - exp_base['pred'].values), 0, None)
                            name = f'{SPECIALIST_BLEND_NAME}__hb{hb:.3f}_h50{h50:.3f}_h100{h100:.3f}_mh{mh:.2f}_lc{lc:.3f}_ml{ml:.2f}'.replace('.', 'p')
                            gs, gb = summarize_specialist_predictions(name, 'groupkfold', group_base['target'].values, gp, group_base['group'].astype(str).values)
                            es, eb = summarize_specialist_predictions(name, 'target_heavy_target_heavy_holdout', exp_base['target'].values, ep, exp_groups)
                            summaries.extend([gs, es])
                            bins.extend([gb, eb])
                            rows.append({'blend_name': name, 'high_base': hb, 'high50_coef': h50, 'high100_coef': h100, 'max_high': mh, 'low_coef': lc, 'max_low': ml, 'groupkfold_mae': float(gs.mae.iloc[0]), 'group_high50_mae': float(gs.high50_mae.iloc[0]), 'group_high100_mae': float(gs.high100_mae.iloc[0]), 'group_pred_mean': float(gs.pred_mean.iloc[0]), 'group_pred_max': float(gs.pred_max.iloc[0]), 'target_heavy_holdout_mae': float(es.mae.iloc[0]), 'target_heavy_holdout_high50_mae': float(es.high50_mae.iloc[0]), 'target_heavy_holdout_high100_mae': float(es.high100_mae.iloc[0]), 'target_heavy_holdout_pred_mean': float(es.pred_mean.iloc[0]), 'target_heavy_holdout_pred_max': float(es.pred_max.iloc[0])})
    grid = pd.DataFrame(rows).sort_values(['groupkfold_mae', 'target_heavy_holdout_mae'])
    grid.to_csv(report_dir / 'specialist_blend_grid.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(report_dir / 'specialist_blend_validation_summary.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(report_dir / 'specialist_blend_bin_report.csv', index=False)
    top = grid.head(20).copy()
    paths = []
    for r in top.itertuples(index=False):
        hw = np.clip(r.high_base + r.high50_coef * test_probs['high50_prob'].values + r.high100_coef * test_probs['high100_prob'].values, 0, r.max_high)
        lw = np.clip(r.low_coef * test_probs['lt1_prob'].values, 0, r.max_low)
        pred = np.clip(base_sub['pred'].values + hw * (high_test - base_sub['pred'].values) + lw * (low_test - base_sub['pred'].values), 0, None)
        sub = pd.read_csv('sample_submission.csv')[['ID']].merge(pd.DataFrame({'ID': test_ids.values, fe_neighbor_feature_missing_exps_TARGET: pred}), on='ID', how='left')
        path = sub_dir / f'{r.blend_name}_submission.csv'
        sub.to_csv(path, index=False)
        paths.append(str(path))
    top['submission_path'] = paths
    top.to_csv(report_dir / 'specialist_blend_top20_groupkfold.csv', index=False)
    return top

def train_high_low_specialist_blend():
    for d in ['reports', 'oof_predictions', 'submissions']:
        (SPECIALIST_BLEND_OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)
    train, test, feature_cols, metadata = fe_context_rolling_features_build_feature_set()
    x, xt = fe_neighbor_feature_missing_exps_fill_features(train, test, feature_cols, 'lag_roll_linear_interpolate')
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    groups = train['scenario_id'].values
    group_base = load_context_quantile_blend_oof('groupkfold')
    exp_base = load_context_quantile_blend_oof('target_heavy_target_heavy_holdout')
    base_sub = load_context_quantile_blend_submission(test['ID'])
    group_probs, exp_probs, test_probs = make_delay_gate_probabilities(train, test, x, xt, y, groups)
    high_oof, high_test, high_folds = train_weighted_specialist_model('high_weighted_lgbm', x, xt, y, groups, make_high_delay_weights(y))
    low_oof, low_test, low_folds = train_weighted_specialist_model('low_weighted_lgbm', x, xt, y, groups, make_low_delay_weights(y))
    high_exp, _, va_mask, scenario_stat = train_specialist_target_heavy_holdout('high_weighted_lgbm', x, y, groups, make_high_delay_weights(y))
    low_exp, _, _, _ = train_specialist_target_heavy_holdout('low_weighted_lgbm', x, y, groups, make_low_delay_weights(y))
    pd.concat([high_folds, low_folds], ignore_index=True).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'reports' / 'specialist_fold_report.csv', index=False)
    scenario_stat.to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'reports' / 'target_heavy_holdout_scenario_target_stat.csv', index=False)
    pd.DataFrame({'target': y.values, 'pred': high_oof, 'group': groups}).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'oof_predictions' / 'high_weighted_lgbm_groupkfold_oof.csv', index=False)
    pd.DataFrame({'target': y.values, 'pred': low_oof, 'group': groups}).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'oof_predictions' / 'low_weighted_lgbm_groupkfold_oof.csv', index=False)
    pd.DataFrame({'ID': test['ID'].values, fe_neighbor_feature_missing_exps_TARGET: high_test}).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'submissions' / 'high_weighted_lgbm_submission.csv', index=False)
    pd.DataFrame({'ID': test['ID'].values, fe_neighbor_feature_missing_exps_TARGET: low_test}).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'submissions' / 'low_weighted_lgbm_submission.csv', index=False)
    summaries, bins = ([], [])
    for name, pred, exp_pred in [('high_weighted_lgbm', high_oof, high_exp), ('low_weighted_lgbm', low_oof, low_exp)]:
        gs, gb = summarize_specialist_predictions(name, 'groupkfold', y.values, pred, groups)
        es, eb = summarize_specialist_predictions(name, 'target_heavy_target_heavy_holdout', y.loc[va_mask].values, exp_pred, groups[va_mask])
        summaries.extend([gs, es])
        bins.extend([gb, eb])
    pd.concat(summaries, ignore_index=True).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'reports' / 'specialist_model_summary.csv', index=False)
    pd.concat(bins, ignore_index=True).to_csv(SPECIALIST_BLEND_OUTPUT_DIR / 'reports' / 'specialist_model_bin_report.csv', index=False)
    top = search_specialist_blend_weights(group_base, exp_base, base_sub, high_oof, low_oof, high_exp, low_exp, high_test, low_test, group_probs, exp_probs, test_probs, groups[va_mask], test['ID'])
    with open(SPECIALIST_BLEND_OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'base_candidate': f'{SPECIALIST_BASE_MODEL_NAME} + {fe_context_rolling_features_WIDE_TAIL_QUANTILE} weight {QUANTILE_BLEND_WEIGHT}', 'plan': 'codex_specialist_blend_specialist_model_plan.md limited scope: high/low weighted LGBM + soft residual blend, no HPO', 'feature_count': len(feature_cols), 'metadata': metadata}, f, ensure_ascii=False, indent=2, default=str)
    fe_experiments_append_history(Path('outputs'), {'experiment_name': SPECIALIST_BLEND_NAME, 'base_after_run': 'context_rolling_context_roll_wide_tail_quantile_blend', 'hypothesis': 'High/low weighted LGBM specialists with OOF soft residual blend, no HPO.', 'feature_count': len(feature_cols), 'groupkfold_mae': float(top.groupkfold_mae.iloc[0]), 'target_heavy_holdout_mae': float(top.target_heavy_holdout_mae.iloc[0]), 'high50_mae': float(top.target_heavy_holdout_high50_mae.iloc[0]), 'high100_mae': float(top.target_heavy_holdout_high100_mae.iloc[0]), 'improved_target_heavy_holdout': bool(top.target_heavy_holdout_mae.iloc[0] < 10.557657), 'best_target_heavy_holdout_after_run': min(float(top.target_heavy_holdout_mae.iloc[0]), 10.557657), 'removed_count': 0})
    print(top.head(12).to_string(index=False), flush=True)

# =============================================================================
# LGBM/XGB/CatBoost 보조 앙상블
# =============================================================================

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
FALLBACK_ENSEMBLE_OUTPUT_DIR = Path('outputs/lgbm_xgb_cat_fallback_ensemble')
FALLBACK_ENSEMBLE_RANDOM_STATE = 42

def inverse_log1p_prediction(pred):
    return np.clip(np.expm1(pred), 0, None)

def apply_low_delay_postprocess_to_best(pred: np.ndarray, lt1_prob: np.ndarray) -> np.ndarray:
    out = np.asarray(pred, dtype=float).copy()
    m = lt1_prob >= 0.7
    out[m] = np.maximum(0, out[m] * 0.8 - 1.0)
    return np.clip(out, 0, None)

def make_lgbm_xgb_cat_models():
    return {'lgbm': lambda seed: lgb.LGBMRegressor(objective='mae', metric='mae', n_estimators=100, learning_rate=0.06, num_leaves=96, min_child_samples=50, subsample=0.9, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=5.0, random_state=seed, n_jobs=-1, verbose=-1), 'xgb': lambda seed: XGBRegressor(objective='reg:absoluteerror', eval_metric='mae', n_estimators=100, learning_rate=0.06, max_depth=7, min_child_weight=8.0, subsample=0.88, colsample_bytree=0.88, reg_alpha=1.2, reg_lambda=5.0, tree_method='hist', random_state=seed, n_jobs=-1, early_stopping_rounds=10), 'cat': lambda seed: CatBoostRegressor(loss_function='MAE', eval_metric='MAE', iterations=100, learning_rate=0.06, depth=7, l2_leaf_reg=5.0, random_seed=seed, allow_writing_files=False, verbose=False, thread_count=-1)}

def reconstruct_best_lb_oof(y, groups):
    context_rolling = pd.read_csv(Path('outputs') / fe_context_rolling_features_BASE_NAME / 'oof_predictions' / f'{fe_context_rolling_features_BASE_NAME}_groupkfold_oof.csv')
    wide_tail_quantile = pd.read_csv('outputs/wide_tail_quantile_lgbm/wide_tail_quantile_alpha_0p55/oof_predictions/wide_tail_quantile_alpha_0p55_groupkfold_oof.csv')
    high = pd.read_csv('outputs/high_low_specialist_blend/oof_predictions/high_weighted_lgbm_groupkfold_oof.csv')['pred'].values
    low = pd.read_csv('outputs/high_low_specialist_blend/oof_predictions/low_weighted_lgbm_groupkfold_oof.csv')['pred'].values
    probs = pd.read_csv('outputs/delay_gate_probabilities/oof_predictions/delay_gate_probability_lgbm_groupkfold_gate_probs.csv')
    base_pred = np.clip(0.675 * context_rolling['pred'].values + 0.325 * wide_tail_quantile['pred'].values, 0, None)
    hw = np.clip(0.6 * probs['high100_prob'].values, 0, 0.35)
    lw = np.clip(0.3 * probs['lt1_prob'].values, 0, 0.3)
    pred = np.clip(base_pred + hw * (high - base_pred) + lw * (low - base_pred), 0, None)
    return apply_low_delay_postprocess_to_best(pred, probs['lt1_prob'].values)

def train_lgbm_xgb_cat_fallback_ensemble():
    for d in ['reports', 'submissions', 'oof_predictions']:
        (FALLBACK_ENSEMBLE_OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)
    print('[BUILD] context_rolling feature store', flush=True)
    train, test, feature_cols, _ = fe_context_rolling_features_build_feature_set()
    imp = pd.read_csv(Path('outputs') / fe_context_rolling_features_BASE_NAME / 'feature_importance' / f'{fe_context_rolling_features_BASE_NAME}_feature_importance_mean.csv')
    cols = [f for f in imp['feature'].tolist() if f in set(feature_cols)][:500]
    x_all, xt_all = fe_neighbor_feature_missing_exps_fill_features(train, test, cols, 'lag_roll_linear_interpolate')
    x = x_all.astype(np.float32)
    xt = xt_all.astype(np.float32)
    y = train[fe_neighbor_feature_missing_exps_TARGET].astype(float)
    y_log = np.log1p(y)
    groups = train['scenario_id'].values
    sample = pd.read_csv('sample_submission.csv')[['ID']]
    oofs = {}
    test_frame = sample.copy()
    rows = []
    for name, maker in make_lgbm_xgb_cat_models().items():
        print(f'[TRAIN] {name}', flush=True)
        oof = np.zeros(len(x))
        test_pred = np.zeros(len(xt))
        for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=5).split(x, y, groups), start=1):
            model = maker(FALLBACK_ENSEMBLE_RANDOM_STATE + fold)
            if name == 'lgbm':
                model.fit(x.iloc[tr_idx], y_log.iloc[tr_idx], eval_set=[(x.iloc[va_idx], y_log.iloc[va_idx])], callbacks=[lgb.early_stopping(10, verbose=False)])
            elif name == 'xgb':
                model.fit(x.iloc[tr_idx], y_log.iloc[tr_idx], eval_set=[(x.iloc[va_idx], y_log.iloc[va_idx])], verbose=False)
            else:
                model.fit(x.iloc[tr_idx], y_log.iloc[tr_idx], eval_set=(x.iloc[va_idx], y_log.iloc[va_idx]), early_stopping_rounds=10, verbose=False)
            oof[va_idx] = inverse_log1p_prediction(model.predict(x.iloc[va_idx]))
            test_pred += inverse_log1p_prediction(model.predict(xt)) / 5
        oofs[name] = oof
        rows.append({'model': name, 'groupkfold_mae': float(np.mean(np.abs(y.values - oof)))})
        pd.DataFrame({'target': y.values, 'pred': oof, 'group': groups}).to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'oof_predictions' / f'{name}_oof.csv', index=False)
        pred_by_id = pd.DataFrame({'ID': test['ID'].values, name: test_pred})
        test_frame = test_frame.merge(pred_by_id, on='ID', how='left')
    best_sub = pd.read_csv('outputs/low_delay_postprocess/submissions/low_delay_postprocess_submission.csv').rename(columns={fe_neighbor_feature_missing_exps_TARGET: 'best'})
    test_frame = test_frame.merge(best_sub, on='ID', how='left')
    best_oof = reconstruct_best_lb_oof(y, groups)
    oofs['best'] = best_oof
    rows.append({'model': 'best_lb_oof_reconstructed', 'groupkfold_mae': float(np.mean(np.abs(y.values - best_oof)))})
    grid_rows = []
    for wb in [0.5, 0.6, 0.7, 0.8]:
        rem = 1.0 - wb
        for wl in [0.0, 0.25, 0.34, 0.5]:
            for wx in [0.0, 0.25, 0.33, 0.5]:
                wc = max(0.0, 1.0 - wl - wx)
                s = wl + wx + wc
                if s == 0:
                    continue
                wl2, wx2, wc2 = (rem * wl / s, rem * wx / s, rem * wc / s)
                pred = wb * best_oof + wl2 * oofs['lgbm'] + wx2 * oofs['xgb'] + wc2 * oofs['cat']
                grid_rows.append({'w_best': wb, 'w_lgbm': wl2, 'w_xgb': wx2, 'w_cat': wc2, 'groupkfold_mae': float(np.mean(np.abs(y.values - pred)))})
    grid = pd.DataFrame(grid_rows).sort_values('groupkfold_mae')
    grid.to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'reports/weight_grid.csv', index=False)
    best = grid.iloc[0]
    test_pred = best.w_best * test_frame['best'].values + best.w_lgbm * test_frame['lgbm'].values + best.w_xgb * test_frame['xgb'].values + best.w_cat * test_frame['cat'].values
    out = sample.copy()
    out[fe_neighbor_feature_missing_exps_TARGET] = np.clip(test_pred, 0, None)
    out.to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'submissions/corrected_weighted_ensemble_submission.csv', index=False)
    mean_pred = test_frame[['best', 'lgbm', 'xgb', 'cat']].mean(axis=1).values
    out = sample.copy()
    out[fe_neighbor_feature_missing_exps_TARGET] = np.clip(mean_pred, 0, None)
    out.to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'submissions/corrected_mean_ensemble_submission.csv', index=False)
    meta_x = pd.DataFrame({k: oofs[k] for k in ['best', 'lgbm', 'xgb', 'cat']})
    meta_t = test_frame[['best', 'lgbm', 'xgb', 'cat']].copy()
    stack_oof = np.zeros(len(meta_x))
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(meta_x, y, groups):
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
        ridge.fit(meta_x.iloc[tr_idx], y.iloc[tr_idx])
        stack_oof[va_idx] = np.clip(ridge.predict(meta_x.iloc[va_idx]), 0, None)
    final = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]))
    final.fit(meta_x, y)
    stack_test = np.clip(final.predict(meta_t), 0, None)
    out = sample.copy()
    out[fe_neighbor_feature_missing_exps_TARGET] = stack_test
    out.to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'submissions/corrected_ridge_stacking_submission.csv', index=False)
    rows.append({'model': 'weighted_grid_ensemble', 'groupkfold_mae': float(best.groupkfold_mae)})
    rows.append({'model': 'corrected_mean_ensemble', 'groupkfold_mae': float(np.mean(np.abs(y.values - meta_x.mean(axis=1).values)))})
    rows.append({'model': 'corrected_ridge_stacking', 'groupkfold_mae': float(np.mean(np.abs(y.values - stack_oof)))})
    report = pd.DataFrame(rows).sort_values('groupkfold_mae')
    report.to_csv(FALLBACK_ENSEMBLE_OUTPUT_DIR / 'reports/corrected_fallback_summary.csv', index=False)
    print(report.to_string(index=False), flush=True)
    print(grid.head(10).to_string(index=False), flush=True)



def prepare_legacy_runtime() -> Path:
    prepare_runtime(SRC_DIR)
    return SRC_DIR


def _apply_low_delay_postprocess(pred: np.ndarray, lt1_prob: np.ndarray) -> np.ndarray:
    out = np.asarray(pred, dtype=float).copy()
    mask = lt1_prob >= 0.70
    out[mask] = np.maximum(0, out[mask] * 0.8 - 1.0)
    return np.clip(out, 0, None)


def _build_low_delay_postprocess_artifact() -> Path:
    top_path = OUTPUTS_DIR / "high_low_specialist_blend" / "reports" / "specialist_blend_top20_groupkfold.csv"
    probs_path = OUTPUTS_DIR / "delay_gate_probabilities" / "oof_predictions" / "delay_gate_probability_lgbm_test_gate_probs.csv"
    if not top_path.exists():
        raise FileNotFoundError(f"Missing specialist_blend top blend report: {top_path}")
    if not probs_path.exists():
        raise FileNotFoundError(f"Missing trained gate probabilities: {probs_path}")
    top = pd.read_csv(top_path)
    if top.empty or "submission_path" not in top.columns:
        raise ValueError(f"Invalid specialist_blend top blend report: {top_path}")
    base_submission_path = Path(str(top.iloc[0]["submission_path"]))
    if not base_submission_path.is_absolute():
        base_submission_path = ROOT / base_submission_path
    base = pd.read_csv(base_submission_path)
    probs = pd.read_csv(probs_path)
    if len(base) != len(probs):
        raise ValueError("specialist_blend top submission and test gate probabilities have different lengths")
    out = base.copy()
    out[TARGET] = _apply_low_delay_postprocess(out[TARGET].to_numpy(dtype=float), probs["lt1_prob"].to_numpy(dtype=float))
    out_dir = OUTPUTS_DIR / "low_delay_postprocess" / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / LOW_DELAY_POSTPROCESS_SUBMISSION_NAME
    out.to_csv(out_path, index=False)
    return out_path


def _run_target_lt1_probability_stage() -> None:
    target_lt1_oof = OUTPUTS_DIR / "target_lt1_probability_features_lgbm" / "oof_predictions" / "target_lt1_probability_features_lgbm_groupkfold_oof.csv"
    if target_lt1_oof.exists():
        print("[1/6] target<1 probability artifacts already exist; skip", flush=True)
        return
    print("[1/6] train target<1 probability/base artifacts", flush=True)
    build_low_delay_probability_outputs()


def _run_wide_tail_quantile_stage() -> None:
    wide_tail_quantile_oof = OUTPUTS_DIR / "wide_tail_quantile_lgbm" / WIDE_TAIL_QUANTILE_MODEL_NAME / "oof_predictions" / f"{WIDE_TAIL_QUANTILE_MODEL_NAME}_groupkfold_oof.csv"
    if wide_tail_quantile_oof.exists():
        print("[2/6] wide_tail_quantile quantile artifacts already exist; skip", flush=True)
        return
    print("[2/6] train wide_tail_quantile quantile artifacts", flush=True)
    train, test, y, groups, feature_sets = build_wide_tail_quantile_features()
    x_q, xt_q, q_cols, q_meta = feature_sets["wide_tail"]
    fe_wide_tail_quantile_lgbm_run_one("wide_tail", 0.55, x_q, xt_q, test, y, groups, q_cols, q_meta)


def _run_context_rolling_model_stage() -> None:
    context_rolling_oof = OUTPUTS_DIR / CONTEXT_ROLLING_MODEL_NAME / "oof_predictions" / f"{CONTEXT_ROLLING_MODEL_NAME}_groupkfold_oof.csv"
    if context_rolling_oof.exists():
        print("[3/6] context_rolling context artifacts already exist; skip", flush=True)
        return
    print("[3/6] train context_rolling context feature model", flush=True)
    train_context_rolling, test_context_rolling, feature_cols, metadata = build_context_rolling_feature_set()
    fe_lgbm_log_target_run_log_target_experiment(
        name=fe_context_rolling_features_BASE_NAME,
        hypothesis="Add non-duplicate backlog proxy, cumulative backlog, and diff x cumulative pressure features to layout/scenario-rank base model.",
        train=train_context_rolling,
        test=test_context_rolling,
        feature_cols=feature_cols,
        metadata=metadata,
    )


def _run_specialist_blend_stage() -> None:
    specialist_top_report = OUTPUTS_DIR / "high_low_specialist_blend" / "reports" / "specialist_blend_top20_groupkfold.csv"
    if specialist_top_report.exists():
        print("[4/6] specialist_blend specialist artifacts already exist; skip", flush=True)
        return
    print("[4/6] train high/low specialist blend", flush=True)
    train_high_low_specialist_blend()


def _run_corrected_fallback_stage() -> Path:
    print("[6/6] train corrected fallback ensemble", flush=True)
    train_lgbm_xgb_cat_fallback_ensemble()
    fallback_submission = OUTPUTS_DIR / "lgbm_xgb_cat_fallback_ensemble" / "submissions" / "corrected_weighted_ensemble_submission.csv"
    if not fallback_submission.exists():
        raise FileNotFoundError(f"Final trained submission was not created: {fallback_submission}")
    return fallback_submission


def run_training_pipeline() -> dict:
    ensure_result_dirs()
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        prepare_legacy_runtime()
        _run_target_lt1_probability_stage()
        _run_wide_tail_quantile_stage()
        _run_context_rolling_model_stage()
        _run_specialist_blend_stage()
        print("[5/6] rebuild low_delay_postprocess low-delay postprocess artifact", flush=True)
        low_delay_postprocess_path = _build_low_delay_postprocess_artifact()
        fallback_submission = _run_corrected_fallback_stage()
        shutil.copyfile(fallback_submission, FINAL_RETRAINED_PATH)
        stats = validate_submission_file(FINAL_RETRAINED_PATH)
        manifest = {
            "submission_path": str(FINAL_RETRAINED_PATH),
            "source_submission_path": str(fallback_submission),
            "rebuilt_low_delay_postprocess_submission_path": str(low_delay_postprocess_path),
            "wide_tail_quantile_model": WIDE_TAIL_QUANTILE_MODEL_NAME,
            "validation_stats": stats,
        }
        with (RESULTS_SUBMISSIONS_DIR / "retraining_manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
        return manifest
    finally:
        os.chdir(old_cwd)


def run_and_compare(reference_path: Path) -> dict:
    result = run_training_pipeline()
    comparison = compare_submission_files(reference_path, FINAL_RETRAINED_PATH)
    pd.DataFrame([comparison]).to_csv(RESULTS_VALIDATION_DIR / "final_retraining_comparison.csv", index=False)
    result["comparison"] = comparison
    return result


def main() -> None:
    result = run_training_pipeline()
    print(f"submission_path: {result['submission_path']}", flush=True)


if __name__ == "__main__":
    main()
