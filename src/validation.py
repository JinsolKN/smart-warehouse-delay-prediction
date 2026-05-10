"""검증 유틸리티와 모델 검증 리포트 함수 모음."""

from __future__ import annotations


# =============================================================================
# 모델 검증 리포트 구현
# =============================================================================

"""
linear_validation_report_utils.py

스마트 창고 출고 지연 예측 프로젝트용 검증 유틸.

목적
----
하나의 모델에 대해 아래 검증 지표를 일관된 형식으로 산출한다.

1. GroupKFold MAE
2. Target-heavy holdout MAE
3. pred max / pred mean / pred quantile
4. target bin별 MAE
5. 50+ / 100+ 구간 예측 평균

권장 사용 상황
-------------
- Linear 재출발: Ridge / Huber / ElasticNet 비교
- target cutoff 실험
- log target 실험
- LGBM 등 비선형 모델의 리스크 검증

주의
----
Target-heavy holdout은 target을 보고 만든 holdout이므로 일반 CV가 아니다.
이 검증셋은 "고지연 리스크 검증셋"으로만 사용한다.
모델 선택은 GroupKFold + Target-heavy holdout을 함께 보고 판단한다.

작성 기준
---------
Target-heavy holdout 설정:
- valid_group_ratio = 0.20
- high_sort_col     = "target_mean"
- high_pool_ratio   = 0.20
- high_group_ratio  = 0.30

사용 예시
---------
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from linear_validation_report_utils import evaluate_model_with_dual_validation

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42))
])

report = evaluate_model_with_dual_validation(
    model=model,
    X=X,
    y=y,
    groups=groups,
    model_name="ridge_pure",
    save_dir="./validation_reports",
    n_splits=5,
    random_state=42,
)

print(report["summary"])
print(report["bin_report"])
"""
import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
val_linear_TARGET_BINS = [-np.inf, 20, 40, 50, 100, np.inf]
val_linear_TARGET_BIN_LABELS = ['<20', '20-40', '40-50', '50-100', '100+']

@dataclass
class val_linear_TargetHeavyConfig:
    """고지연 검증용 target-heavy holdout 설정."""
    valid_group_ratio: float = 0.2
    high_sort_col: str = 'target_mean'
    high_pool_ratio: float = 0.2
    high_group_ratio: float = 0.3
    random_state: int = 42

def val_linear_high_delay_weight_func(y_train):
    w = np.ones(len(y_train))
    w[(y_train >= 40) & (y_train < 50)] = 3.0
    w[(y_train >= 50) & (y_train < 100)] = 5.0
    w[y_train >= 100] = 8.0
    return w

def val_linear__ensure_numpy(x: Any) -> np.ndarray:
    """pandas Series, list, numpy array를 모두 numpy array로 변환합니다."""
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, pd.Series):
        return x.values
    return np.asarray(x)

def val_linear__safe_predict(model, X: pd.DataFrame, clip_zero: bool=True) -> np.ndarray:
    """모델 예측 후 필요하면 0 이하 예측을 0으로 clip."""
    pred = model.predict(X)
    pred = np.asarray(pred, dtype=float)
    if clip_zero:
        pred = np.clip(pred, 0, None)
    return pred

def val_linear_make_prediction_frame(y_true: np.ndarray, y_pred: np.ndarray, groups: Optional[np.ndarray]=None) -> pd.DataFrame:
    """예측 결과 DataFrame 생성."""
    y_true = val_linear__ensure_numpy(y_true).astype(float)
    y_pred = val_linear__ensure_numpy(y_pred).astype(float)
    out = pd.DataFrame({'target': y_true, 'pred': y_pred, 'error': y_pred - y_true, 'abs_error': np.abs(y_pred - y_true)})
    if groups is not None:
        out['group'] = val_linear__ensure_numpy(groups)
    out['target_bin'] = pd.cut(out['target'], bins=val_linear_TARGET_BINS, labels=val_linear_TARGET_BIN_LABELS)
    return out

def val_linear_summarize_prediction_frame(pred_df: pd.DataFrame, validation_name: str, model_name: str) -> pd.DataFrame:
    """전체 MAE, pred max, 50+/100+ 예측 평균 등을 요약."""
    y_true = pred_df['target'].values
    y_pred = pred_df['pred'].values
    mask_50 = y_true >= 50
    mask_100 = y_true >= 100
    summary = {'model_name': model_name, 'validation': validation_name, 'n_rows': len(pred_df), 'mae': mean_absolute_error(y_true, y_pred), 'target_mean': float(np.mean(y_true)), 'target_median': float(np.median(y_true)), 'target_max': float(np.max(y_true)), 'pred_mean': float(np.mean(y_pred)), 'pred_median': float(np.median(y_pred)), 'pred_max': float(np.max(y_pred)), 'pred_p90': float(np.quantile(y_pred, 0.9)), 'pred_p95': float(np.quantile(y_pred, 0.95)), 'pred_p99': float(np.quantile(y_pred, 0.99)), 'high50_count': int(mask_50.sum()), 'high50_ratio': float(mask_50.mean()), 'high50_target_mean': float(np.mean(y_true[mask_50])) if mask_50.any() else np.nan, 'high50_pred_mean': float(np.mean(y_pred[mask_50])) if mask_50.any() else np.nan, 'high50_mae': float(np.mean(np.abs(y_true[mask_50] - y_pred[mask_50]))) if mask_50.any() else np.nan, 'high100_count': int(mask_100.sum()), 'high100_ratio': float(mask_100.mean()), 'high100_target_mean': float(np.mean(y_true[mask_100])) if mask_100.any() else np.nan, 'high100_pred_mean': float(np.mean(y_pred[mask_100])) if mask_100.any() else np.nan, 'high100_mae': float(np.mean(np.abs(y_true[mask_100] - y_pred[mask_100]))) if mask_100.any() else np.nan}
    return pd.DataFrame([summary])

def val_linear_make_bin_report(pred_df: pd.DataFrame, validation_name: str, model_name: str) -> pd.DataFrame:
    """타깃 구간별 건수, 실제 평균, 예측 평균, MAE를 계산합니다."""
    report = pred_df.groupby('target_bin', observed=False).agg(count=('target', 'count'), target_mean=('target', 'mean'), pred_mean=('pred', 'mean'), mae=('abs_error', 'mean'), pred_max=('pred', 'max'), target_max=('target', 'max')).reset_index()
    report.insert(0, 'validation', validation_name)
    report.insert(0, 'model_name', model_name)
    return report

def val_linear_evaluate_groupkfold(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name: str, n_splits: int=5, random_state: int=42, clip_zero: bool=True, sample_weight_func=None) -> Dict[str, pd.DataFrame]:
    """
    GroupKFold OOF 검증 수행.

    반환:
    - summary: 전체 요약
    - fold_report: fold별 결과
    - bin_report: target bin별 결과
    - oof: row별 OOF 예측
    """
    y = val_linear__ensure_numpy(y).astype(float)
    groups = val_linear__ensure_numpy(groups)
    splitter = GroupKFold(n_splits=n_splits)
    oof_pred = np.zeros(len(X), dtype=float)
    fold_rows: List[Dict[str, Any]] = []
    for fold, (tr_idx, va_idx) in enumerate(tqdm(splitter.split(X, y, groups), total=n_splits, desc=f'{model_name} | GroupKFold')):
        X_tr = X.iloc[tr_idx]
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y[va_idx]
        m = clone(model)
        if sample_weight_func is not None:
            fit_weight = sample_weight_func(y_tr)
            m.fit(X_tr, y_tr, sample_weight=fit_weight)
        else:
            m.fit(X_tr, y_tr)
        pred = val_linear__safe_predict(m, X_va, clip_zero=clip_zero)
        oof_pred[va_idx] = pred
        fold_rows.append({'model_name': model_name, 'validation': 'groupkfold', 'fold': fold, 'n_train': len(tr_idx), 'n_valid': len(va_idx), 'n_train_groups': len(np.unique(groups[tr_idx])), 'n_valid_groups': len(np.unique(groups[va_idx])), 'valid_target_mean': float(np.mean(y_va)), 'valid_target_max': float(np.max(y_va)), 'pred_mean': float(np.mean(pred)), 'pred_max': float(np.max(pred)), 'mae': float(mean_absolute_error(y_va, pred))})
    oof = val_linear_make_prediction_frame(y, oof_pred, groups=groups)
    summary = val_linear_summarize_prediction_frame(oof, 'groupkfold', model_name)
    fold_report = pd.DataFrame(fold_rows)
    bin_report = val_linear_make_bin_report(oof, 'groupkfold', model_name)
    return {'summary': summary, 'fold_report': fold_report, 'bin_report': bin_report, 'oof': oof}

def val_linear_make_scenario_target_stat(y: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """시나리오별 타깃 통계를 생성합니다."""
    y = val_linear__ensure_numpy(y).astype(float)
    groups = val_linear__ensure_numpy(groups)
    tmp = pd.DataFrame({'group': groups, 'target': y})
    stat = tmp.groupby('group').agg(n_rows=('target', 'count'), target_mean=('target', 'mean'), target_median=('target', 'median'), target_max=('target', 'max'), target_std=('target', 'std'), high50_ratio=('target', lambda s: np.mean(s >= 50)), high100_ratio=('target', lambda s: np.mean(s >= 100)), high200_ratio=('target', lambda s: np.mean(s >= 200))).reset_index()
    stat['target_std'] = stat['target_std'].fillna(0)
    return stat

def val_linear_make_target_heavy_holdout_split(y: np.ndarray, groups: np.ndarray, config: Optional[TargetHeavyConfig]=None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Target-heavy holdout split 생성.

    반환:
    - train_mask
    - valid_mask
    - scenario_stat
    """
    if config is None:
        config = val_linear_TargetHeavyConfig()
    y = val_linear__ensure_numpy(y).astype(float)
    groups = val_linear__ensure_numpy(groups)
    rng = np.random.default_rng(config.random_state)
    scenario_stat = val_linear_make_scenario_target_stat(y, groups)
    stat = scenario_stat.sort_values(config.high_sort_col, ascending=False).reset_index(drop=True)
    all_groups = stat['group'].values
    n_total_groups = len(all_groups)
    n_valid_groups = int(n_total_groups * config.valid_group_ratio)
    n_high_pool = int(n_total_groups * config.high_pool_ratio)
    high_pool = stat.head(n_high_pool)['group'].values
    normal_pool = stat.iloc[n_high_pool:]['group'].values
    n_high_valid = int(n_valid_groups * config.high_group_ratio)
    n_normal_valid = n_valid_groups - n_high_valid
    n_high_valid = min(n_high_valid, len(high_pool))
    n_normal_valid = min(n_normal_valid, len(normal_pool))
    valid_high_groups = rng.choice(high_pool, size=n_high_valid, replace=False)
    valid_normal_groups = rng.choice(normal_pool, size=n_normal_valid, replace=False)
    valid_groups = set(np.r_[valid_high_groups, valid_normal_groups])
    valid_mask = np.array([g in valid_groups for g in groups])
    train_mask = ~valid_mask
    return (train_mask, valid_mask, scenario_stat)

def val_linear_evaluate_target_heavy_holdout(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name: str, config: Optional[TargetHeavyConfig]=None, clip_zero: bool=True, sample_weight_func=None) -> Dict[str, pd.DataFrame]:
    """
    Target-heavy holdout holdout 검증 수행.

    반환:
    - summary: 전체 요약
    - fold_report: holdout 설정 및 결과
    - bin_report: target bin별 결과
    - oof: valid row별 예측
    - scenario_stat: scenario별 target 통계
    """
    if config is None:
        config = val_linear_TargetHeavyConfig()
    y = val_linear__ensure_numpy(y).astype(float)
    groups = val_linear__ensure_numpy(groups)
    train_mask, valid_mask, scenario_stat = val_linear_make_target_heavy_holdout_split(y, groups, config)
    X_tr = X.loc[train_mask]
    y_tr = y[train_mask]
    X_va = X.loc[valid_mask]
    y_va = y[valid_mask]
    groups_va = groups[valid_mask]
    m = clone(model)
    if sample_weight_func is not None:
        fit_weight = sample_weight_func(y_tr)
        m.fit(X_tr, y_tr, sample_weight=fit_weight)
    else:
        m.fit(X_tr, y_tr)
    pred = val_linear__safe_predict(m, X_va, clip_zero=clip_zero)
    valid_pred_df = val_linear_make_prediction_frame(y_va, pred, groups=groups_va)
    summary = val_linear_summarize_prediction_frame(valid_pred_df, 'target_heavy_holdout', model_name)
    for key, value in asdict(config).items():
        summary[key] = value
    fold_report = pd.DataFrame([{'model_name': model_name, 'validation': 'target_heavy_holdout', 'n_train': int(train_mask.sum()), 'n_valid': int(valid_mask.sum()), 'n_train_groups': len(np.unique(groups[train_mask])), 'n_valid_groups': len(np.unique(groups[valid_mask])), 'valid_target_mean': float(np.mean(y_va)), 'valid_target_median': float(np.median(y_va)), 'valid_target_max': float(np.max(y_va)), 'valid_high50_ratio': float(np.mean(y_va >= 50)), 'valid_high100_ratio': float(np.mean(y_va >= 100)), 'pred_mean': float(np.mean(pred)), 'pred_max': float(np.max(pred)), 'mae': float(mean_absolute_error(y_va, pred)), **asdict(config)}])
    bin_report = val_linear_make_bin_report(valid_pred_df, 'target_heavy_holdout', model_name)
    return {'summary': summary, 'fold_report': fold_report, 'bin_report': bin_report, 'oof': valid_pred_df, 'scenario_stat': scenario_stat}

def val_linear_evaluate_model_with_dual_validation(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, model_name: str, save_dir: Optional[str]=None, n_splits: int=5, random_state: int=42, target_heavy_config: Optional[TargetHeavyConfig]=None, clip_zero: bool=True, sample_weight_func=None) -> Dict[str, pd.DataFrame]:
    """
    GroupKFold + Target-heavy holdout을 한 번에 평가.

    저장 파일:
    - {model_name}_summary.csv
    - {model_name}_fold_report.csv
    - {model_name}_bin_report.csv
    - {model_name}_groupkfold_oof.csv
    - {model_name}_target_heavy_holdout_oof.csv
    - {model_name}_scenario_target_stat.csv
    - {model_name}_config.json
    """
    if target_heavy_config is None:
        target_heavy_config = val_linear_TargetHeavyConfig(random_state=random_state)
    group_result = val_linear_evaluate_groupkfold(model=model, X=X, y=y, groups=groups, model_name=model_name, n_splits=n_splits, random_state=random_state, clip_zero=clip_zero, sample_weight_func=sample_weight_func)
    target_heavy_result = val_linear_evaluate_target_heavy_holdout(model=model, X=X, y=y, groups=groups, model_name=model_name, config=target_heavy_config, clip_zero=clip_zero, sample_weight_func=sample_weight_func)
    summary = pd.concat([group_result['summary'], target_heavy_result['summary']], ignore_index=True)
    fold_report = pd.concat([group_result['fold_report'], target_heavy_result['fold_report']], ignore_index=True, sort=False)
    bin_report = pd.concat([group_result['bin_report'], target_heavy_result['bin_report']], ignore_index=True)
    result = {'summary': summary, 'fold_report': fold_report, 'bin_report': bin_report, 'groupkfold_oof': group_result['oof'], 'target_heavy_holdout_oof': target_heavy_result['oof'], 'scenario_target_stat': target_heavy_result['scenario_stat']}
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        summary.to_csv(os.path.join(save_dir, f'{model_name}_summary.csv'), index=False)
        fold_report.to_csv(os.path.join(save_dir, f'{model_name}_fold_report.csv'), index=False)
        bin_report.to_csv(os.path.join(save_dir, f'{model_name}_bin_report.csv'), index=False)
        result['groupkfold_oof'].to_csv(os.path.join(save_dir, f'{model_name}_groupkfold_oof.csv'), index=False)
        result['target_heavy_holdout_oof'].to_csv(os.path.join(save_dir, f'{model_name}_target_heavy_holdout_oof.csv'), index=False)
        result['scenario_target_stat'].to_csv(os.path.join(save_dir, f'{model_name}_scenario_target_stat.csv'), index=False)
        config_payload = {'model_name': model_name, 'n_splits': n_splits, 'random_state': random_state, 'clip_zero': clip_zero, 'target_heavy_config': asdict(target_heavy_config), 'target_bins': val_linear_TARGET_BINS, 'target_bin_labels': val_linear_TARGET_BIN_LABELS}
        with open(os.path.join(save_dir, f'{model_name}_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_payload, f, ensure_ascii=False, indent=2)
    return result

def val_linear_evaluate_many_models_with_dual_validation(models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, save_dir: Optional[str]=None, n_splits: int=5, random_state: int=42, target_heavy_config: Optional[TargetHeavyConfig]=None, clip_zero: bool=True) -> Dict[str, Any]:
    """
    여러 모델을 동일한 기준으로 평가.

    models 예시:
    models = {
        "ridge": ridge_model,
        "huber": huber_model,
        "elasticnet": elasticnet_model,
    }
    """
    all_results: Dict[str, Any] = {}
    summary_list = []
    bin_list = []
    fold_list = []
    for model_name, model in models.items():
        result = val_linear_evaluate_model_with_dual_validation(model=model, X=X, y=y, groups=groups, model_name=model_name, save_dir=save_dir, n_splits=n_splits, random_state=random_state, target_heavy_config=target_heavy_config, clip_zero=clip_zero)
        all_results[model_name] = result
        summary_list.append(result['summary'])
        bin_list.append(result['bin_report'])
        fold_list.append(result['fold_report'])
    combined_summary = pd.concat(summary_list, ignore_index=True)
    combined_bin_report = pd.concat(bin_list, ignore_index=True)
    combined_fold_report = pd.concat(fold_list, ignore_index=True, sort=False)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        combined_summary.to_csv(os.path.join(save_dir, 'combined_model_summary.csv'), index=False)
        combined_bin_report.to_csv(os.path.join(save_dir, 'combined_model_bin_report.csv'), index=False)
        combined_fold_report.to_csv(os.path.join(save_dir, 'combined_model_fold_report.csv'), index=False)
    all_results['combined_summary'] = combined_summary
    all_results['combined_bin_report'] = combined_bin_report
    all_results['combined_fold_report'] = combined_fold_report
    return all_results

# =============================================================================
# 제출 파일 검증/비교
# =============================================================================


from pathlib import Path

import numpy as np
import pandas as pd

from config import SAMPLE_SUBMISSION_PATH, TARGET


def load_submission(path: str | Path) -> pd.DataFrame:
    """제출 CSV 파일을 읽습니다."""
    return pd.read_csv(path)


def validate_submission_frame(frame: pd.DataFrame) -> dict[str, float]:
    """제출 파일의 형태, ID 순서, 결측, 예측값 범위를 검사합니다."""
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if list(frame.columns) != ["ID", TARGET]:
        raise ValueError(f"Submission columns must be ['ID', {TARGET!r}], got {list(frame.columns)!r}")
    if len(frame) != len(sample):
        raise ValueError(f"Submission row count mismatch: {len(frame)} != {len(sample)}")
    if not frame["ID"].equals(sample["ID"]):
        raise ValueError("Submission ID order differs from sample_submission.csv")
    if frame[TARGET].isna().any():
        raise ValueError("Submission contains missing predictions")
    if (frame[TARGET] < 0).any():
        raise ValueError("Submission contains negative predictions")
    pred = frame[TARGET].astype(float)
    return {
        "rows": float(len(frame)),
        "pred_mean": float(pred.mean()),
        "pred_p95": float(pred.quantile(0.95)),
        "pred_max": float(pred.max()),
    }


def validate_submission_file(path: str | Path) -> dict[str, float]:
    """제출 파일을 검증하고 요약 통계를 반환합니다."""
    return validate_submission_frame(load_submission(path))


def compare_submission_frames(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | bool]:
    """검증된 두 제출 파일을 row 단위로 비교합니다."""
    validate_submission_frame(reference)
    validate_submission_frame(candidate)
    if not reference["ID"].equals(candidate["ID"]):
        raise ValueError("Reference and candidate ID order differs")
    ref = reference[TARGET].to_numpy(dtype=float)
    cand = candidate[TARGET].to_numpy(dtype=float)
    diff = cand - ref
    return {
        "exact_equal": bool(np.array_equal(ref, cand)),
        "allclose_1e_12": bool(np.allclose(ref, cand, rtol=0, atol=1e-12)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "num_diff_gt_1e_12": int(np.sum(np.abs(diff) > 1e-12)),
        "reference_pred_mean": float(np.mean(ref)),
        "candidate_pred_mean": float(np.mean(cand)),
    }


def compare_submission_files(reference_path: str | Path, candidate_path: str | Path) -> dict[str, float | bool]:
    """기준 제출 파일과 후보 제출 파일을 읽고 비교합니다."""
    return compare_submission_frames(load_submission(reference_path), load_submission(candidate_path))
