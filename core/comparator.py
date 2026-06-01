from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from core.f0_extractor import F0Result


@dataclass
class SyllableComparison:
    syllable_idx: int
    native_f0: np.ndarray        # (n_frames,) resampled z-score, 0 where unvoiced
    learner_f0: np.ndarray       # (n_frames,) resampled z-score, 0 where unvoiced
    joint_voiced_mask: np.ndarray  # (n_frames,) bool: native AND learner both voiced
    native_duration: float       # seconds
    learner_duration: float      # seconds


class IntonationComparator:
    def compare(
        self,
        native: F0Result,
        learner: F0Result,
        native_boundaries: list[tuple[float, float]],
        learner_boundaries: list[tuple[float, float]],
        n_frames: int = 50,
    ) -> list[SyllableComparison]:
        results = []
        for idx, ((n_start, n_end), (l_start, l_end)) in enumerate(
            zip(native_boundaries, learner_boundaries)
        ):
            native_f0, native_voiced = self._resample_syllable(native, n_start, n_end, n_frames)
            learner_f0, learner_voiced = self._resample_syllable(learner, l_start, l_end, n_frames)

            results.append(SyllableComparison(
                syllable_idx=idx,
                native_f0=native_f0,
                learner_f0=learner_f0,
                joint_voiced_mask=native_voiced & learner_voiced,
                native_duration=n_end - n_start,
                learner_duration=l_end - l_start,
            ))
        return results

    def _resample_syllable(
        self,
        f0_result: F0Result,
        t_start: float,
        t_end: float,
        n_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = (f0_result.times >= t_start) & (f0_result.times < t_end)
        f0 = f0_result.f0[mask] # 음절 단위로 normalized pitch 추출
        voiced = f0_result.voiced_mask[mask]

        if len(f0) == 0:
            return np.zeros(n_frames), np.zeros(n_frames, dtype=bool)

        src = np.linspace(0, n_frames - 1, len(f0))
        dst = np.arange(n_frames, dtype=float)
        resampled_f0 = np.interp(dst, src, f0)
        resampled_voiced = np.interp(dst, src, voiced.astype(float)) > 0.5
        return resampled_f0, resampled_voiced