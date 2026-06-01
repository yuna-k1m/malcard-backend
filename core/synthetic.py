"""Phase 0 검증용 synthetic perturbation 생성기.

[vector-level]
분포에서 정상 벡터를 known 방향으로 이동시켜 분류기 sensitivity를 검증한다.

[audio-level]
TTS wav에 parselmouth(F0) + librosa(time stretch) perturbation 주입.
Phase 2 러시아어권 test sample 생성용.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import parselmouth
from parselmouth.praat import call as _praat

from core.eojeol_vector import DIM_NAMES
from core.distribution import GaussianEojeolDistribution

PerturbationKind = Literal["rising", "falling", "flat", "elongation", "slow"]

# 각 kind가 classify()에서 기대하는 label
EXPECTED_LABEL: dict[str, str] = {
    "rising":      "rising 과도",
    "falling":     "falling 과도",
    "flat":        "억양 평탄",
    "elongation":  "마지막 음절 elongation",
    "slow":        "어절 전체 느림",
}

_DIM_IDX = {name: i for i, name in enumerate(DIM_NAMES)}


def perturb_vector(
    base: np.ndarray,
    dist: GaussianEojeolDistribution,
    kind: PerturbationKind,
    magnitude: float = 3.5,
) -> tuple[np.ndarray, str]:
    """기저 벡터에 perturbation을 주입해 known-label 벡터를 반환.

    Args:
        base: 정상 벡터 (12,). 분포 안쪽에 있어야 함.
        dist: 이미 fit()된 분포. mean/std 기준으로 perturbation 크기 결정.
        kind: 변형 종류.
        magnitude: perturbation 크기 (σ 단위).

    Returns:
        (perturbed_vector, expected_label) 튜플.
    """
    v = base.copy()
    std = dist.std_

    if kind == "rising":
        v[_DIM_IDX["f0_slope"]] = dist.mean_[_DIM_IDX["f0_slope"]] + magnitude * std[_DIM_IDX["f0_slope"]]

    elif kind == "falling":
        v[_DIM_IDX["f0_slope"]] = dist.mean_[_DIM_IDX["f0_slope"]] - magnitude * std[_DIM_IDX["f0_slope"]]

    elif kind == "flat":
        # slope을 평균 근방으로 고정 + f0_range를 -magnitude*σ
        slope_idx = _DIM_IDX["f0_slope"]
        range_idx = _DIM_IDX["f0_range"]
        v[slope_idx] = dist.mean_[slope_idx]                            # |z_slope| < 0.5 보장
        v[range_idx] = dist.mean_[range_idx] - magnitude * std[range_idx]

    elif kind == "elongation":
        idx = _DIM_IDX["last_syl_ratio"]
        v[idx] = dist.mean_[idx] + magnitude * std[idx]

    elif kind == "slow":
        idx = _DIM_IDX["duration"]
        v[idx] = dist.mean_[idx] + magnitude * std[idx]

    return v, EXPECTED_LABEL[kind]


def generate_test_cases(
    dist: GaussianEojeolDistribution,
    n_per_class: int = 10,
    magnitude: float = 3.5,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, str]]:
    """각 perturbation 종류별 n_per_class개의 (vector, label) 쌍 생성.

    base 벡터는 분포 mean 주변 0.3σ 내 랜덤 샘플 → 실제 검증 다양성 확보.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    cases: list[tuple[np.ndarray, str]] = []
    for kind in EXPECTED_LABEL:
        for _ in range(n_per_class):
            # 분포 내부의 무작위 정상 벡터에서 시작
            base = dist.mean_ + rng.standard_normal(_N_DIMS) * dist.std_ * 0.3
            v, label = perturb_vector(base, dist, kind, magnitude=magnitude)  # type: ignore[arg-type]
            cases.append((v, label))

    return cases


_N_DIMS = len(DIM_NAMES)

# ── Audio-level perturbation ─────────────────────────────────────────────────
# parselmouth PSOLA 재합성으로 F0 조작, librosa로 시간축 조작.
# 반환값: (samples: float32 ndarray, sample_rate: int)

_MANIP_TS = 0.01   # pitch analysis time step (s)
_MIN_F0 = 75.0     # Hz
_MAX_F0 = 600.0    # Hz


def _make_manip(wav_path: Path | str):
    sound = parselmouth.Sound(str(wav_path))
    manip = _praat(sound, "To Manipulation", _MANIP_TS, _MIN_F0, _MAX_F0)
    return sound, manip


def perturb_audio_f0_shift(
    wav_path: Path | str,
    semitones: float = 6.0,
) -> tuple[np.ndarray, int]:
    """전체 발화 F0를 semitones만큼 이동 후 PSOLA 재합성.

    semitones > 0 → 음높이 상승 (rising 과도 시뮬레이션)
    semitones < 0 → 음높이 하강
    """
    _, manip = _make_manip(wav_path)
    pt = _praat(manip, "Extract pitch tier")
    t0 = _praat(pt, "Get start time")
    t1 = _praat(pt, "Get end time")
    _praat(pt, "Shift frequencies", t0, t1, semitones, "semitones")
    _praat([pt, manip], "Replace pitch tier")
    out = _praat(manip, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32), int(out.sampling_frequency)


def perturb_audio_slope_flip(
    wav_path: Path | str,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, int]:
    """지정 구간의 F0 contour를 시간축으로 반전 (rising → falling).

    t_start/t_end 미지정 시 전체 발화에 적용.
    """
    _, manip = _make_manip(wav_path)
    pt = _praat(manip, "Extract pitch tier")
    ta0 = _praat(pt, "Get start time")
    ta1 = _praat(pt, "Get end time")
    ts = ta0 if t_start is None else t_start
    te = ta1 if t_end is None else t_end

    n = int(_praat(pt, "Get number of points"))
    pts: list[tuple[float, float]] = [
        (_praat(pt, "Get time from index", i), _praat(pt, "Get value at index", i))
        for i in range(1, n + 1)
    ]

    # 타겟 범위 안의 pitch point만 시간 반전, 주파수는 유지
    mirrored = [
        (ts + (te - t), f) if ts <= t <= te else (t, f)
        for t, f in pts
    ]
    mirrored.sort(key=lambda x: x[0])

    _praat(pt, "Remove points between", ta0, ta1)
    for t, f in mirrored:
        _praat(pt, "Add point", t, f)

    _praat([pt, manip], "Replace pitch tier")
    out = _praat(manip, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32), int(out.sampling_frequency)


def perturb_audio_f0_ramp(
    wav_path: Path | str,
    semitones_start: float = 0.0,
    semitones_end: float = 8.0,
) -> tuple[np.ndarray, int]:
    """발화 시작→끝으로 F0를 선형 ramp (PSOLA 재합성).

    f0_slope, f0_end dim을 분포에서 이탈시켜 'rising 과도' 감지.
    semitones_start=0, semitones_end=8 → 끝 부분이 +8 semitone 올라감.
    """
    _, manip = _make_manip(wav_path)
    pt = _praat(manip, "Extract pitch tier")
    t0 = _praat(pt, "Get start time")
    t1 = _praat(pt, "Get end time")
    n = int(_praat(pt, "Get number of points"))
    pts: list[tuple[float, float]] = [
        (_praat(pt, "Get time from index", i), _praat(pt, "Get value at index", i))
        for i in range(1, n + 1)
    ]
    new_pts = []
    for t, f in pts:
        pos = (t - t0) / (t1 - t0) if t1 > t0 else 0.5
        semitones = semitones_start + (semitones_end - semitones_start) * pos
        factor = 2.0 ** (semitones / 12.0)
        new_pts.append((t, f * factor))
    _praat(pt, "Remove points between", t0, t1)
    for t, f in new_pts:
        _praat(pt, "Add point", t, f)
    _praat([pt, manip], "Replace pitch tier")
    out = _praat(manip, "Get resynthesis (overlap-add)")
    return out.values[0].astype(np.float32), int(out.sampling_frequency)


def perturb_audio_elongate_last(
    wav_path: Path | str,
    last_syl_start: float,
    factor: float = 2.0,
    target_sr: int = 16000,
    min_samples: int = 4096,
) -> tuple[np.ndarray, int]:
    """마지막 음절 구간을 factor배 시간 연장 (librosa time_stretch).

    last_syl_start: 마지막 음절 시작 시간(초). 이 이후가 늘어남.
    min_samples: 구간이 너무 짧으면 onset을 앞당겨 최소 길이 확보.
    """
    import librosa
    y, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    onset = int(last_syl_start * sr)
    # 구간이 너무 짧으면 onset을 앞당겨 n_fft 제한 회피
    if len(y) - onset < min_samples:
        onset = max(0, len(y) - min_samples)
    stretched = librosa.effects.time_stretch(y[onset:], rate=1.0 / factor)
    return np.concatenate([y[:onset], stretched]).astype(np.float32), sr


def save_audio(samples: np.ndarray, sr: int, out_path: Path | str) -> Path:
    """float32 numpy → WAV 파일 저장."""
    import soundfile as sf
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), samples, sr, subtype="PCM_16")
    return out