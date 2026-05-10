from __future__ import annotations

from config import FINAL_REFERENCE_PATH
from pipeline import run_pipeline


# =============================================================================
# 제출 파일 생성
# =============================================================================


def build_final_submission() -> dict:
    """전체 파이프라인을 실행하고 최종 제출 메타데이터를 반환합니다."""
    return run_pipeline(FINAL_REFERENCE_PATH)


# =============================================================================
# 실행
# =============================================================================


def main() -> None:
    result = build_final_submission()
    print(f"submission_path: {result['submission_path']}", flush=True)
    print("comparison:", result["comparison"], flush=True)


if __name__ == "__main__":
    main()
