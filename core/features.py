"""신호 feature 축. 무엇을 비교 신호로 쓸지 (f0 / delta-f0).

numpy·scipy 외 의존성 없음. Segmenter/Aligner와 직교하는 세 번째 축.
새 feature는 `F0Result -> np.ndarray` 함수 하나 추가하면 끝.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from core.f0_extractor import F0Result


def interp_unvoiced(f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """무성 구간을 인접 유성값으로 선형 보간. 양끝은 상수 외삽.

    DTW(모양)와 delta(속도)가 공유하는 단일 진실원 — 무성/blank를
    어떻게 잇느냐가 두 경로에서 갈리면 안 된다.
    """
    f0 = f0.astype(float)
    vi = np.where(voiced)[0]
    if len(vi) == 0:
        return f0
    return np.interp(np.arange(len(f0)), vi, f0[vi])


def f0_feature(fr: F0Result) -> np.ndarray:
    """기본 feature: z-score f0 그대로."""
    return fr.f0


def delta_f0(
    fr: F0Result,
    window: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    """피치 변화 속도. 보간 → Savitzky-Golay 스무딩+1차 미분 (한 번에).

    무성/blank를 먼저 보간해 차분 스파이크를 제거하고(= DTW와 동일 처리),
    savgol(deriv=1)로 트래커 jitter를 누른 '지각되는 억양 속도'를 낸다.
    판정용 아님 — 탐색 뷰 전용. 길이/타이밍 판정은 vector+distribution.
    """
    c = interp_unvoiced(fr.f0, fr.voiced_mask)
    n = len(c)
    if n < 3:
        return np.zeros(n)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 3:
        win = 3
    poly = min(polyorder, win - 1)
    return savgol_filter(c, win, poly, deriv=1)


def slice_signal(
    times: np.ndarray, arr: np.ndarray, voiced: np.ndarray,
    t0: float, t1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """시간 구간 [t0, t1)으로 신호 슬라이스. (arr, voiced, times) 반환."""
    mask = (times >= t0) & (times < t1)
    return arr[mask], voiced[mask], times[mask]
