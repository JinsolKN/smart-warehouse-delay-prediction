##전체 학습 파이프라인 실행 파일.

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import (
    CONTEXT_ROLLING_MODEL_NAME,
    LOW_DELAY_POSTPROCESS_SUBMISSION_NAME,
    FINAL_REFERENCE_PATH,
    FINAL_RETRAINED_PATH,
    OUTPUTS_DIR,
    WIDE_TAIL_QUANTILE_MODEL_NAME,
    RESULTS_SUBMISSIONS_DIR,
    RESULTS_VALIDATION_DIR,
    ROOT,
    ensure_result_dirs,
)
from feature_engineering import (
    build_low_delay_probability_outputs,
    build_context_rolling_feature_set,
    build_wide_tail_quantile_features,
    fe_wide_tail_quantile_lgbm_run_one,
    fe_context_rolling_features_BASE_NAME,
    fe_lgbm_log_target_exps_run_log_target_experiment,
)
from modeling import (
    _build_low_delay_postprocess_artifact,
    train_high_low_specialist_blend as train_high_low_specialist_blend_model,
    train_lgbm_xgb_cat_fallback_ensemble as train_lgbm_xgb_cat_fallback_ensemble_model,
    prepare_legacy_runtime,
)
from validation import compare_submission_files, validate_submission_file


@dataclass
class PipelineArtifacts:
    target_lt1_oof: Path
    wide_tail_quantile_oof: Path
    context_rolling_oof: Path
    specialist_blend_top_report: Path
    low_delay_postprocess_submission: Path
    fallback_ensemble_submission: Path
    final_submission: Path


def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}", flush=True)


def artifact_paths() -> PipelineArtifacts:
    return PipelineArtifacts(
        target_lt1_oof=(
            OUTPUTS_DIR
            / "target_lt1_probability_features_lgbm"
            / "oof_predictions"
            / "target_lt1_probability_features_lgbm_groupkfold_oof.csv"
        ),
        wide_tail_quantile_oof=(
            OUTPUTS_DIR
            / "wide_tail_quantile_lgbm"
            / WIDE_TAIL_QUANTILE_MODEL_NAME
            / "oof_predictions"
            / f"{WIDE_TAIL_QUANTILE_MODEL_NAME}_groupkfold_oof.csv"
        ),
        context_rolling_oof=(
            OUTPUTS_DIR
            / CONTEXT_ROLLING_MODEL_NAME
            / "oof_predictions"
            / f"{CONTEXT_ROLLING_MODEL_NAME}_groupkfold_oof.csv"
        ),
        specialist_blend_top_report=(
            OUTPUTS_DIR
            / "high_low_specialist_blend"
            / "reports"
            / "specialist_blend_top20_groupkfold.csv"
        ),
        low_delay_postprocess_submission=(
            OUTPUTS_DIR
            / "low_delay_postprocess"
            / "submissions"
            / LOW_DELAY_POSTPROCESS_SUBMISSION_NAME
        ),
        fallback_ensemble_submission=(
            OUTPUTS_DIR
            / "lgbm_xgb_cat_fallback_ensemble"
            / "submissions"
            / "corrected_weighted_ensemble_submission.csv"
        ),
        final_submission=FINAL_RETRAINED_PATH,
    )


# =============================================================================
# 0. 실행 환경 준비
# =============================================================================


def setup_runtime() -> None:
    """결과 폴더를 만들고 실험 모듈을 import할 수 있게 준비합니다."""
    ensure_result_dirs()
    prepare_legacy_runtime()


# =============================================================================
# 1. 데이터 전처리 / 피처 산출물 생성
# =============================================================================


def build_low_delay_preprocessing(paths: PipelineArtifacts) -> None:
    """target<1 확률 피처와 기본 검증 산출물을 생성합니다."""
    print_section("1. 데이터 전처리 - 저지연 확률 피처")

    if paths.target_lt1_oof.exists():
        print(f"skip: {paths.target_lt1_oof}", flush=True)
        return

    build_low_delay_probability_outputs()


def build_quantile_feature_set(paths: PipelineArtifacts) -> None:
    """고지연 tail 보완용 피처셋과 0.55 분위수 모델 산출물을 생성합니다."""
    print_section("2. 피처셋 - 고지연 tail quantile 입력")

    if paths.wide_tail_quantile_oof.exists():
        print(f"skip: {paths.wide_tail_quantile_oof}", flush=True)
        return

    train, test, y, groups, feature_sets = build_wide_tail_quantile_features()
    x_q, xt_q, q_cols, q_meta = feature_sets["wide_tail"]

    print(f"train rows: {len(train):,}", flush=True)
    print(f"test rows: {len(test):,}", flush=True)
    print(f"wide_tail features: {len(q_cols):,}", flush=True)

    fe_wide_tail_quantile_lgbm_run_one("wide_tail", 0.55, x_q, xt_q, test, y, groups, q_cols, q_meta)


# =============================================================================
# 2. 모델링
# =============================================================================


def run_context_rolling_context_model_stage(paths: PipelineArtifacts) -> None:
    """주력 LightGBM log-target context/rolling 모델을 학습합니다."""
    print_section("3. 모델링 - Context Rolling 모델")

    if paths.context_rolling_oof.exists():
        print(f"skip: {paths.context_rolling_oof}", flush=True)
        return

    train, test, feature_cols, metadata = build_context_rolling_feature_set()
    print(f"train rows: {len(train):,}", flush=True)
    print(f"test rows: {len(test):,}", flush=True)
    print(f"features: {len(feature_cols):,}", flush=True)

    fe_lgbm_log_target_exps_run_log_target_experiment(
        name=fe_context_rolling_features_BASE_NAME,
        hypothesis=(
            "Add non-duplicate backlog proxy, cumulative backlog, and diff x "
            "cumulative pressure features to layout/scenario-rank base model."
        ),
        train=train,
        test=test,
        feature_cols=feature_cols,
        metadata=metadata,
    )


def run_high_low_specialist_blend_stage(paths: PipelineArtifacts) -> None:
    """고지연/저지연 구간을 각각 보완하는 specialist 모델을 학습합니다."""
    print_section("4. 모델링 - 고지연/저지연 Specialist Blend")

    if paths.specialist_blend_top_report.exists():
        print(f"skip: {paths.specialist_blend_top_report}", flush=True)
        return

    train_high_low_specialist_blend_model()


def rebuild_low_delay_postprocess(paths: PipelineArtifacts) -> Path:
    """저지연 보정이 적용된 postprocess 제출 산출물을 다시 생성합니다."""
    print_section("5. 모델링 - 저지연 Postprocess")

    low_delay_postprocess_path = _build_low_delay_postprocess_artifact()
    print(f"low_delay_postprocess submission: {low_delay_postprocess_path}", flush=True)
    return low_delay_postprocess_path


def run_lgbm_xgb_cat_fallback_ensemble_stage(paths: PipelineArtifacts) -> None:
    """LGBM/XGB/CatBoost fallback 모델과 가중 앙상블을 학습합니다."""
    print_section("6. 모델링 - 보정 Fallback Ensemble")

    if paths.fallback_ensemble_submission.exists():
        print(f"skip: {paths.fallback_ensemble_submission}", flush=True)
        return

    train_lgbm_xgb_cat_fallback_ensemble_model()
    if not paths.fallback_ensemble_submission.exists():
        raise FileNotFoundError(f"missing final trained submission: {paths.fallback_ensemble_submission}")


# =============================================================================
# 3. 검증
# =============================================================================


def validate_final_submission(paths: PipelineArtifacts, reference_path: Path) -> dict:
    """새 제출 파일을 검증하고, 기준 제출 파일이 있으면 예측 차이도 비교합니다."""
    print_section("8. 검증")

    stats = validate_submission_file(paths.final_submission)
    RESULTS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    print("submission stats:", stats, flush=True)

    if not reference_path.exists():
        print(f"reference comparison skipped: {reference_path} not found", flush=True)
        return {"stats": stats, "comparison": None}

    comparison = compare_submission_files(reference_path, paths.final_submission)
    pd.DataFrame([comparison]).to_csv(
        RESULTS_VALIDATION_DIR / "final_retraining_comparison.csv",
        index=False,
    )
    print("reference comparison:", comparison, flush=True)
    return {"stats": stats, "comparison": comparison}


# =============================================================================
# 4. 제출 파일 생성
# =============================================================================


def write_final_submission(paths: PipelineArtifacts, low_delay_postprocess_path: Path) -> dict:
    """선택된 앙상블 제출 파일을 results/submissions 경로로 복사합니다."""
    print_section("7. 제출 파일 생성")

    RESULTS_SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.fallback_ensemble_submission, paths.final_submission)

    manifest = {
        "submission_path": str(paths.final_submission),
        "source_submission_path": str(paths.fallback_ensemble_submission),
        "rebuilt_low_delay_postprocess_submission_path": str(low_delay_postprocess_path),
        "wide_tail_quantile_model": WIDE_TAIL_QUANTILE_MODEL_NAME,
    }
    manifest_path = RESULTS_SUBMISSIONS_DIR / "retraining_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"final submission: {paths.final_submission}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    return manifest


def run_pipeline(reference_path: Path = FINAL_REFERENCE_PATH) -> dict:
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        paths = artifact_paths()

        setup_runtime()
        build_low_delay_preprocessing(paths)
        build_quantile_feature_set(paths)
        run_context_rolling_context_model_stage(paths)
        run_high_low_specialist_blend_stage(paths)
        low_delay_postprocess_path = rebuild_low_delay_postprocess(paths)
        run_lgbm_xgb_cat_fallback_ensemble_stage(paths)
        manifest = write_final_submission(paths, low_delay_postprocess_path)
        validation = validate_final_submission(paths, reference_path)

        manifest.update(validation)
        return manifest
    finally:
        os.chdir(old_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="전체 학습 파이프라인.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=FINAL_REFERENCE_PATH,
        help="최종 비교에 사용할 기준 제출 파일.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.reference)
    print_section("완료")
    print(f"submission_path: {result['submission_path']}", flush=True)
    print("comparison:", result["comparison"], flush=True)


if __name__ == "__main__":
    main()
