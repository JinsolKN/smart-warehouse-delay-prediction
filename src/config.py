## 프로젝트 공통 설정 파일.


from __future__ import annotations

from pathlib import Path


# =============================================================================
# 경로 설정
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
LAYOUT_PATH = RAW_DATA_DIR / "layout_info.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
LEGACY_SAMPLE_SUBMISSION_PATH = ROOT / "sample_submission.csv"


# =============================================================================
# 산출물 저장 위치
# =============================================================================

OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
RESULTS_SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
RESULTS_VALIDATION_DIR = RESULTS_DIR / "validation"


# =============================================================================
# 학습 설정
# =============================================================================

TARGET = "avg_delay_minutes_next_30m"
RANDOM_STATE = 42


# =============================================================================
# 제출/비교 파일
# =============================================================================

FINAL_REFERENCE_PATH = ROOT / "final_submission.csv"
FINAL_RETRAINED_PATH = RESULTS_SUBMISSIONS_DIR / "final_submission_retrained.csv"


# =============================================================================
# 주요 실험명
# =============================================================================

CONTEXT_ROLLING_MODEL_NAME = "context_rolling_lgbm"
WIDE_TAIL_QUANTILE_MODEL_NAME = "wide_tail_quantile_alpha_0p55"
LOW_DELAY_POSTPROCESS_SUBMISSION_NAME = "low_delay_postprocess_submission.csv"


def ensure_result_dirs() -> None:
    """학습, 검증, 제출 단계에서 사용할 결과 폴더를 생성합니다."""
    for path in [OUTPUTS_DIR, RESULTS_DIR, RESULTS_SUBMISSIONS_DIR, RESULTS_VALIDATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
