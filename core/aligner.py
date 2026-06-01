"""정렬 축. (native_slice, learner_slice) → 같은 길이 곡선 2개 + x.

forced_alignment를 모른다. z-score f0 배열 한 쌍만 받는다.
새 정렬 방식은 여기 클래스 하나 추가하면 끝 — lens/plotter/segmenter 무관.
"""
from __future__ import annotations

import numpy as np
from dtaidistance import dtw

from core.features import interp_unvoiced


class NoAligner:
    """정렬 없음. 각 구간을 시간 비례로 n_frames에 선형 리샘플(워핑 X)."""

    def __init__(self, n_frames: int = 50):
        self.n_frames = n_frames

    def align(
        self,
        n_f0: np.ndarray, n_voiced: np.ndarray,
        l_f0: np.ndarray, l_voiced: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n_frames
        x = np.arange(n)
        nc, nv = _resample(n_f0, n_voiced, n)
        lc, lv = _resample(l_f0, l_voiced, n)
        return x, nc, lc, nv & lv


class DtwAligner:
    """DTW 워핑(모양 전용). 무성 구간 보간 → 연속 contour → warping path.

    타이밍/길이는 워프가 정규화해 사라진다. 길이 오류 판정은 이 렌즈가
    아니라 eojeol_vector(duration/last_syl_ratio) + distribution의 몫.
    """

    def __init__(self, n_frames: int = 50):
        self.n_frames = n_frames  # 빈 구간 fallback 길이

    def align(
        self,
        n_f0: np.ndarray, n_voiced: np.ndarray,
        l_f0: np.ndarray, l_voiced: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(n_f0) == 0 or len(l_f0) == 0:
            z = np.zeros(self.n_frames)
            return np.arange(self.n_frames), z, z, np.zeros(self.n_frames, dtype=bool)

        # 무성/blank를 보간해 연속 contour로 만든 뒤 워프 (모양 전용).
        # 0 평탄대끼리의 무의미한 정렬로 워프가 망가지는 걸 방지한다.
        # plot 곡선은 보간값이지만, 메트릭은 lens에서 joint_voiced(원래 유성)
        # 프레임만 사용하므로 보간값이 판정 수치를 오염시키지 않는다.
        n_interp = interp_unvoiced(n_f0, n_voiced)
        l_interp = interp_unvoiced(l_f0, l_voiced)
        path = dtw.warping_path(n_interp, l_interp)
        ni = np.array([i for i, _ in path])
        li = np.array([j for _, j in path])
        x = np.arange(len(path))
        return x, n_interp[ni], l_interp[li], n_voiced[ni] & l_voiced[li]


def _resample(
    f0: np.ndarray, voiced: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(f0) == 0:
        return np.zeros(n), np.zeros(n, dtype=bool)
    src = np.linspace(0, n - 1, len(f0))
    dst = np.arange(n, dtype=float)
    rf = np.interp(dst, src, f0)
    rv = np.interp(dst, src, voiced.astype(float)) > 0.5
    return rf, rv
