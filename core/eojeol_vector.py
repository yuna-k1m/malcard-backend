"""어절 단위 12-dim prosody vector 추출."""
from __future__ import annotations

import numpy as np

from core.f0_extractor import F0Result

DIM_NAMES: list[str] = [
    "f0_mean",        # voiced z-score F0 평균
    "f0_std",         # voiced z-score F0 표준편차
    "f0_slope",       # 어절 시작→끝 기울기 (정규화 시간축 기준)
    "f0_range",       # voiced F0 max - min
    "f0_max_pos",     # 최고점 상대 위치 (0~1)
    "f0_min_pos",     # 최저점 상대 위치 (0~1)
    "f0_start",       # 첫 voiced frame F0
    "f0_end",         # 마지막 voiced frame F0
    "duration",       # 어절 길이 (초)
    "syllable_count", # 음절 수
    "last_syl_ratio", # 마지막 음절 길이 / 평균 음절 길이
    "voiced_ratio",   # voiced frame / total frame
]

_N_DIMS = len(DIM_NAMES)


def extract_eojeol_vector(
    f0_result: F0Result,
    eojeol_boundary: tuple[float, float],
    syllable_boundaries: list[tuple[float, float]],
) -> np.ndarray:
    """어절 한 개의 12-dim prosody vector 반환.

    Args:
        f0_result: 전체 발화 F0 (z-score 정규화 완료).
        eojeol_boundary: (t_start, t_end) in seconds.
        syllable_boundaries: 이 어절 안의 음절 경계 리스트.

    Returns:
        shape (12,) float64 vector. voiced frame 부족 시 해당 dim은 0 또는 기본값.
    """
    t_start, t_end = eojeol_boundary
    mask = (f0_result.times >= t_start) & (f0_result.times < t_end)

    f0_z = f0_result.f0[mask]
    voiced = f0_result.voiced_mask[mask]
    total_frames = len(f0_z)

    voiced_indices = np.where(voiced)[0]
    voiced_f0 = f0_z[voiced]
    voiced_count = len(voiced_f0)

    duration = float(t_end - t_start)

    # ── F0 통계 ──────────────────────────────────────────────────────────────
    if voiced_count >= 2:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_range = float(voiced_f0.max() - voiced_f0.min())
        f0_start = float(voiced_f0[0])
        f0_end = float(voiced_f0[-1])

        norm_pos = voiced_indices / max(total_frames - 1, 1)
        f0_max_pos = float(norm_pos[np.argmax(voiced_f0)])
        f0_min_pos = float(norm_pos[np.argmin(voiced_f0)])

        if voiced_count >= 3:
            f0_slope = float(np.polyfit(norm_pos, voiced_f0, 1)[0])
        else:
            f0_slope = float(f0_end - f0_start)
    else:
        f0_mean = f0_std = f0_slope = f0_range = 0.0
        f0_max_pos = f0_min_pos = 0.5
        f0_start = f0_end = 0.0

    # ── Duration 통계 ────────────────────────────────────────────────────────
    syllable_count = len(syllable_boundaries)

    if syllable_count >= 2:
        syl_durs = [b[1] - b[0] for b in syllable_boundaries]
        avg_dur = float(np.mean(syl_durs))
        last_syl_ratio = syl_durs[-1] / avg_dur if avg_dur > 0 else 1.0
    else:
        last_syl_ratio = 1.0

    voiced_ratio = float(voiced_count / total_frames) if total_frames > 0 else 0.0

    return np.array([
        f0_mean, f0_std, f0_slope, f0_range,
        f0_max_pos, f0_min_pos, f0_start, f0_end,
        duration, float(syllable_count), float(last_syl_ratio), voiced_ratio,
    ], dtype=np.float64)