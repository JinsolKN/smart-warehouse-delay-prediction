from __future__ import annotations

import shutil
import sys

import pandas as pd

from config import (
    LAYOUT_PATH,
    LEGACY_SAMPLE_SUBMISSION_PATH,
    ROOT,
    SAMPLE_SUBMISSION_PATH,
    SRC_DIR,
    TEST_PATH,
    TRAIN_PATH,
)


def configure_import_paths(extra_path=None) -> None:
    """``src``와 선택적으로 전달된 추가 경로를 ``sys.path``에 등록합니다."""
    for path in [SRC_DIR, extra_path]:
        if path is None:
            continue
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def ensure_raw_data_paths() -> None:
    """필수 대회 데이터 파일이 없으면 실행을 중단합니다."""
    required = [TRAIN_PATH, TEST_PATH, LAYOUT_PATH, SAMPLE_SUBMISSION_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")


def ensure_legacy_sample_submission() -> None:
    """루트 경로의 sample_submission.csv가 필요한 코드와 호환되도록 파일을 맞춥니다."""
    if not LEGACY_SAMPLE_SUBMISSION_PATH.exists():
        shutil.copyfile(SAMPLE_SUBMISSION_PATH, LEGACY_SAMPLE_SUBMISSION_PATH)


def prepare_runtime(extra_path=None) -> None:
    """실험 실행 전에 import 경로와 파일 호환성을 준비합니다."""
    configure_import_paths(extra_path)
    ensure_raw_data_paths()
    ensure_legacy_sample_submission()


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """노트북에서 사용하던 순서와 동일하게 train, test, layout 데이터를 읽습니다."""
    ensure_raw_data_paths()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    layout = pd.read_csv(LAYOUT_PATH)
    return train, test, layout


def load_sample_submission() -> pd.DataFrame:
    """대회 sample submission 파일을 읽습니다."""
    return pd.read_csv(SAMPLE_SUBMISSION_PATH)


def load_reference_submission() -> pd.DataFrame:
    """저장되어 있는 기준 제출 파일 final_submission.csv를 읽습니다."""
    from config import FINAL_REFERENCE_PATH

    return pd.read_csv(FINAL_REFERENCE_PATH)
