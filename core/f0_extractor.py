from __future__ import annotations

from pathlib import Path

import numpy as np
import parselmouth
from dataclasses import dataclass


@dataclass
class F0Result:
    f0: np.ndarray        # z-score 정규화된 f0
    f0_raw: np.ndarray    # 원본 Hz 값
    voiced_mask: np.ndarray
    times: np.ndarray
    sampling_rate: float

    @property
    def voiced(self) -> np.ndarray:
        return self.f0[self.voiced_mask]

    @property
    def voiced_raw(self) -> np.ndarray:
        return self.f0_raw[self.voiced_mask]

    @property
    def voiced_count(self) -> int:
        return int(self.voiced_mask.sum())

    @property
    def total_frames(self) -> int:
        return len(self.f0)


def _normalize_f0_zscore(f0: np.ndarray) -> np.ndarray:
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return np.zeros_like(f0, dtype=float)
    std = float(np.std(voiced))
    if std == 0.0:
        return np.zeros_like(f0, dtype=float)
    mean = float(np.mean(voiced))
    f0_norm = np.zeros_like(f0, dtype=float)
    f0_norm[f0 > 0] = (f0[f0 > 0] - mean) / std
    return f0_norm


def extract_f0(wav_path: str | Path) -> F0Result:
    snd = parselmouth.Sound(str(wav_path))
    snd_resampled = snd.resample(new_frequency=16000, precision=50)
    pitch = snd_resampled.to_pitch()

    times = pitch.xs()
    f0_raw = pitch.selected_array['frequency']
    voiced_mask = f0_raw > 0
    f0 = _normalize_f0_zscore(f0_raw)

    return F0Result(
        f0=f0,
        f0_raw=f0_raw,
        voiced_mask=voiced_mask,
        times=times,
        sampling_rate=snd_resampled.sampling_frequency,
    )