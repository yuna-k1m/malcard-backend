"""F0/F1/F2 시각화 — prosody 외 *모음 정체성* 차원 탐색.

설계(grill 합의):
- F0: z-score normalize + interp_unvoiced 연속 곡선 (plot_eojeol_dtw와 통일)
- F1, F2: raw Hz (절대 모음 정체성 보존, 학습자 직관) + NaN interp (gap 제거)
- DTW alignment: F1+F2 z-score multivariate. F0는 그 path 좌표 위에 lookup
  → *모음 정체성 시간*에 F0가 어떻게 흘러가는지 본다 (기존 F0-자체 alignment
  lens와 redundant 의도 — 다른 시점에서 같은 데이터).
- record/rule 무관, 시각화 전용.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parselmouth
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.f0_extractor import F0Result
from core.features import interp_unvoiced

SAMPLE_RATE = 16000
NATIVE_COLOR = "#185FA5"
LEARNER_COLOR = "#993C1D"


@dataclass
class FormantResult:
    f1: np.ndarray        # raw Hz, unvoiced/invalid = NaN
    f2: np.ndarray
    times: np.ndarray
    voiced_mask: np.ndarray

    def slice(self, t0: float, t1: float) -> FormantResult:
        """시간 구간 [t0, t1)으로 슬라이스."""
        mask = (self.times >= t0) & (self.times < t1)
        return FormantResult(
            f1=self.f1[mask], f2=self.f2[mask],
            times=self.times[mask], voiced_mask=self.voiced_mask[mask],
        )


def extract_formants(wav_path: str | Path) -> FormantResult:
    snd = parselmouth.Sound(str(wav_path)).resample(new_frequency=SAMPLE_RATE, precision=50)
    formant = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5)
    times = np.array(formant.xs())
    f1 = np.array([formant.get_value_at_time(1, t) for t in times])
    f2 = np.array([formant.get_value_at_time(2, t) for t in times])
    f1 = np.where(np.isnan(f1) | (f1 <= 0), np.nan, f1)
    f2 = np.where(np.isnan(f2) | (f2 <= 0), np.nan, f2)
    voiced = ~(np.isnan(f1) | np.isnan(f2))
    return FormantResult(f1=f1, f2=f2, times=times, voiced_mask=voiced)


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """NaN을 인접 유효값으로 선형 보간."""
    valid = ~np.isnan(x)
    if not valid.any():
        return np.zeros_like(x)
    vi = np.where(valid)[0]
    return np.interp(np.arange(len(x)), vi, x[vi])


def _zscore(x: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(x)
    if not valid.any():
        return np.zeros_like(x)
    mean = float(np.nanmean(x))
    std = float(np.nanstd(x))
    if std == 0:
        return x - mean
    return (x - mean) / std



def _align_dtw(learner: FormantResult, native: FormantResult) -> np.ndarray | None:
    """F1+F2 z-score multivariate DTW. 반환: warping path [(l_idx, n_idx), ...] (시간 순)."""
    if learner.f1.size < 2 or native.f1.size < 2:
        return None
    l_norm = np.stack([_zscore(_interp_nan(learner.f1)), _zscore(_interp_nan(learner.f2))])
    n_norm = np.stack([_zscore(_interp_nan(native.f1)), _zscore(_interp_nan(native.f2))])
    from librosa.sequence import dtw
    _, wp = dtw(X=l_norm, Y=n_norm, metric="euclidean")
    return wp[::-1]


def f0_zscore_at_times(f0r: F0Result, query_times: np.ndarray) -> np.ndarray:
    """z-score f0 + interp_unvoiced 연속 곡선 → query 시간에 가장 가까운 idx의 값.

    plot_eojeol_dtw와 동일한 처리: unvoiced 구간 보간으로 메운 연속 신호.
    mfcc.py도 같은 lookup 패턴 사용 (path는 다르지만 F0 처리는 통일).
    """
    if len(f0r.times) == 0 or len(query_times) == 0:
        return np.zeros(len(query_times))
    continuous = interp_unvoiced(f0r.f0, f0r.voiced_mask)
    idx = np.searchsorted(f0r.times, query_times)
    idx = np.clip(idx, 0, len(f0r.times) - 1)
    return continuous[idx]


def _pair_traces(
    l_vals: np.ndarray, n_vals: np.ndarray, y_label: str, unit: str, fmt: str = ".0f",
) -> tuple[go.Scatter, go.Scatter]:
    """이미 path 좌표 정렬된 두 신호 → learner / native trace."""
    x = np.arange(len(l_vals))
    return (
        go.Scatter(
            x=x, y=l_vals, mode="lines", name=f"학습자 {y_label}",
            line=dict(color=LEARNER_COLOR, width=2),
            hovertemplate=f"step=%{{x}}<br>{y_label}=%{{y:{fmt}}} {unit}<extra></extra>",
        ),
        go.Scatter(
            x=x, y=n_vals, mode="lines", name=f"원어민 {y_label}",
            line=dict(color=NATIVE_COLOR, width=2),
            hovertemplate=f"step=%{{x}}<br>{y_label}=%{{y:{fmt}}} {unit}<extra></extra>",
        ),
    )


def _signals_along_path(
    learner: FormantResult, native: FormantResult, wp: np.ndarray,
    learner_f0: F0Result, native_f0: F0Result,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """F1+F2 DTW path 좌표 위에 F0(z-score+interp) / F1, F2(raw Hz + NaN interp) 정렬."""
    l_idx, n_idx = wp[:, 0], wp[:, 1]
    l_t = learner.times[l_idx]
    n_t = native.times[n_idx]
    l_f0 = f0_zscore_at_times(learner_f0, l_t)
    n_f0 = f0_zscore_at_times(native_f0, n_t)
    l_f1 = _interp_nan(learner.f1)[l_idx]
    n_f1 = _interp_nan(native.f1)[n_idx]
    l_f2 = _interp_nan(learner.f2)[l_idx]
    n_f2 = _interp_nan(native.f2)[n_idx]
    return l_f0, n_f0, l_f1, n_f1, l_f2, n_f2


_SIGNAL_SPEC = [
    ("F0", "z-score", ".2f"),
    ("F1", "Hz", ".0f"),
    ("F2", "Hz", ".0f"),
]


def _add_f0f1f2_to_column(
    fig: go.Figure, learner: FormantResult, native: FormantResult, wp: np.ndarray,
    learner_f0: F0Result, native_f0: F0Result,
    row_f0: int, row_f1: int, row_f2: int, col: int,
) -> None:
    """한 col에 세 row(F0/F1/F2)로 trace 추가."""
    l_f0, n_f0, l_f1, n_f1, l_f2, n_f2 = _signals_along_path(
        learner, native, wp, learner_f0, native_f0,
    )
    for (vals_l, vals_n), (label, unit, fmt), row in (
        ((l_f0, n_f0), _SIGNAL_SPEC[0], row_f0),
        ((l_f1, n_f1), _SIGNAL_SPEC[1], row_f1),
        ((l_f2, n_f2), _SIGNAL_SPEC[2], row_f2),
    ):
        lt, nt = _pair_traces(vals_l, vals_n, label, unit, fmt)
        fig.add_traces([lt, nt], rows=[row, row], cols=[col, col])


def _add_f0f1f2_to_row(
    fig: go.Figure, learner: FormantResult, native: FormantResult, wp: np.ndarray,
    learner_f0: F0Result, native_f0: F0Result,
    row: int, col_f0: int, col_f1: int, col_f2: int,
) -> None:
    """한 row에 세 col(F0/F1/F2)로 trace 추가."""
    l_f0, n_f0, l_f1, n_f1, l_f2, n_f2 = _signals_along_path(
        learner, native, wp, learner_f0, native_f0,
    )
    for (vals_l, vals_n), (label, unit, fmt), col in (
        ((l_f0, n_f0), _SIGNAL_SPEC[0], col_f0),
        ((l_f1, n_f1), _SIGNAL_SPEC[1], col_f1),
        ((l_f2, n_f2), _SIGNAL_SPEC[2], col_f2),
    ):
        lt, nt = _pair_traces(vals_l, vals_n, label, unit, fmt)
        fig.add_traces([lt, nt], rows=[row, row], cols=[col, col])


def build_global_figure(
    learner: FormantResult, native: FormantResult,
    learner_f0: F0Result, native_f0: F0Result,
    title: str = "F0/F1/F2 (발화 전체, F1+F2 DTW 정렬)",
) -> go.Figure:
    wp = _align_dtw(learner, native)
    if wp is None or len(wp) == 0:
        return go.Figure().update_layout(title=f"{title} — 데이터 없음")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("F0 (피치, z-score)", "F1 (입 열림, Hz)", "F2 (혀 전후, Hz)"),
        vertical_spacing=0.08,
    )
    _add_f0f1f2_to_column(fig, learner, native, wp, learner_f0, native_f0, 1, 2, 3, 1)
    fig.update_yaxes(title_text="F0 (z-score)", row=1, col=1)
    fig.update_yaxes(title_text="F1 (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="F2 (Hz)", row=3, col=1)
    fig.update_xaxes(title_text="F1+F2 DTW path step", row=3, col=1)
    fig.update_layout(title=title, template="plotly_white", height=800, hovermode="x unified")
    return fig


def build_eojeol_figure(
    learner: FormantResult, native: FormantResult,
    learner_f0: F0Result, native_f0: F0Result,
    eojeol_native_spans: list[tuple[float, float]],
    eojeol_learner_spans: list[tuple[float, float]],
    eojeol_labels: list[str],
    title: str = "F0/F1/F2 (어절별, F1+F2 DTW 정렬)",
) -> go.Figure:
    n_eo = min(len(eojeol_native_spans), len(eojeol_learner_spans))
    if n_eo == 0:
        return go.Figure().update_layout(title=f"{title} — 어절 없음")

    subtitles: list[str] = []
    for i in range(n_eo):
        label = eojeol_labels[i] if i < len(eojeol_labels) else f"어절 {i}"
        subtitles.extend([f"{label} — F0 (z)", f"{label} — F1 (Hz)", f"{label} — F2 (Hz)"])

    fig = make_subplots(
        rows=n_eo, cols=3, subplot_titles=subtitles,
        vertical_spacing=0.05, horizontal_spacing=0.06,
    )
    for i in range(n_eo):
        l_slice = learner.slice(*eojeol_learner_spans[i])
        n_slice = native.slice(*eojeol_native_spans[i])
        wp = _align_dtw(l_slice, n_slice)
        if wp is None or len(wp) == 0:
            continue
        _add_f0f1f2_to_row(fig, l_slice, n_slice, wp, learner_f0, native_f0,
                           row=i + 1, col_f0=1, col_f1=2, col_f2=3)
        fig.update_yaxes(title_text="z-score", row=i + 1, col=1)
        fig.update_yaxes(title_text="Hz", row=i + 1, col=2)
        fig.update_yaxes(title_text="Hz", row=i + 1, col=3)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="DTW step", row=n_eo, col=c)
    fig.update_layout(
        title=title, template="plotly_white",
        height=max(300, 240 * n_eo), showlegend=False,
    )
    return fig
