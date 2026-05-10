from __future__ import annotations

import pandas as pd

from config import RESULTS_VALIDATION_DIR, TARGET, TRAIN_PATH, ensure_result_dirs


# =============================================================================
# EDA - target 분포 확인
# =============================================================================


def save_eda_snapshot() -> dict:
    """빠른 확인을 위해 타깃 분포 요약을 저장합니다."""
    ensure_result_dirs()
    train = pd.read_csv(TRAIN_PATH, usecols=[TARGET])
    out = RESULTS_VALIDATION_DIR / "target_distribution_snapshot.csv"
    train[TARGET].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_csv(out)
    return {"target_distribution": out}


# =============================================================================
# 실행
# =============================================================================


def main() -> None:
    print(save_eda_snapshot(), flush=True)


if __name__ == "__main__":
    main()
