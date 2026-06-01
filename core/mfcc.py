"""MFCC + CMVN 시각화 — articulation alignment 비교 (prosody 외 채널).

설계 (grill 합의):
- c1~c12 multivariate DTW로 alignment (c0=energy 제외, delta/ΔΔ 없음 — 시각화 전용).
- per-utterance CMVN(평균/표준편차)으로 화자/채널 차이 제거 → no-norm vs CMVN 2 plot.
- F0는 MFCC DTW path 위에 lookup (formants.f0_zscore_at_times 재사용 — 처리 통일).
- record/rule 무관. global only (어절/음절 slicing 안 함 — normalization은 per-utterance).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.f0_extractor import F0Result
from core.formants import f0_zscore_at_times

SAMPLE_RATE = 16000
N_MFCC = 13  # c0~c12. c0(energy)는 DTW/diff에서 제외.
NATIVE_COLOR = "#185FA5"
LEARNER_COLOR = "#993C1D"


@dataclass
class MfccResult:
    mfcc: np.ndarray   # (n_mfcc, n_frames)
    times: np.ndarray  # (n_frames,)


def extract_mfcc(wav_path: str | Path, hop_length: int = 160) -> MfccResult:
    """librosa로 MFCC 추출. hop_length=160 → 10ms frame @ 16kHz."""
    import librosa
    audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length)
    return MfccResult(mfcc=mfcc, times=times)


def _cmvn(mfcc: np.ndarray) -> np.ndarray:
    """per-utterance Cepstral Mean-Variance Normalization (계수별)."""
    mean = mfcc.mean(axis=1, keepdims=True)
    std = mfcc.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (mfcc - mean) / std


def _align_dtw(learner: np.ndarray, native: np.ndarray) -> np.ndarray | None:
    """c1~c12 multivariate DTW. 반환: warping path [(l_idx, n_idx), ...] (시간 순)."""
    l = learner[1:]  # skip c0
    n = native[1:]
    if l.shape[1] < 2 or n.shape[1] < 2:
        return None
    from librosa.sequence import dtw
    _, wp = dtw(X=l, Y=n, metric="euclidean")
    return wp[::-1]


def build_global_figure(
    learner: MfccResult,
    native: MfccResult,
    learner_f0: F0Result,
    native_f0: F0Result,
    normalize: bool,
    title: str | None = None,
) -> go.Figure:
    """row1=F0 lines (learner/native), row2=(learner − native) MFCC diff heatmap.

    둘 다 MFCC DTW path step x축 공유. normalize=True면 CMVN 후 DTW.
    """
    title = title or ("MFCC + F0 (CMVN)" if normalize else "MFCC + F0 (no-norm)")

    l_mfcc = _cmvn(learner.mfcc) if normalize else learner.mfcc
    n_mfcc = _cmvn(native.mfcc) if normalize else native.mfcc

    wp = _align_dtw(l_mfcc, n_mfcc)
    if wp is None or len(wp) == 0:
        return go.Figure().update_layout(title=f"{title} — 데이터 없음")

    l_idx, n_idx = wp[:, 0], wp[:, 1]
    l_t = learner.times[l_idx]
    n_t = native.times[n_idx]

    l_f0 = f0_zscore_at_times(learner_f0, l_t)
    n_f0 = f0_zscore_at_times(native_f0, n_t)

    diff = l_mfcc[1:, l_idx] - n_mfcc[1:, n_idx]
    x = np.arange(len(wp))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("F0 (z-score)", "MFCC c1~c12 diff (학습자 − 원어민)"),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.6],
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=l_f0, mode="lines", name="학습자 F0",
            line=dict(color=LEARNER_COLOR, width=2),
            hovertemplate="step=%{x}<br>F0=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=n_f0, mode="lines", name="원어민 F0",
            line=dict(color=NATIVE_COLOR, width=2),
            hovertemplate="step=%{x}<br>F0=%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=diff,
            x=x,
            y=np.arange(1, N_MFCC),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="diff", len=0.55, y=0.27),
            hovertemplate="step=%{x}<br>c%{y}<br>diff=%{z:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="F0 (z-score)", row=1, col=1)
    fig.update_yaxes(title_text="MFCC coefficient", row=2, col=1, dtick=1)
    fig.update_xaxes(title_text="MFCC DTW path step", row=2, col=1)
    fig.update_layout(title=title, template="plotly_white", height=700, hovermode="x unified")
    return fig


def build_prosody_plot_data(
    learner: MfccResult,
    native: MfccResult,
    learner_f0: F0Result,
    native_f0: F0Result,
    eojeol_learner_spans: list[tuple[float, float]],
    eojeol_labels: list[str],
) -> dict:
    """CMVN MFCC DTW path 위 F0 lookup + 어절 boundary path step → UI plot data dict.

    UI는 path step x축에 learner_f0/native_f0를 line으로 그리고,
    eojeol_boundaries의 path_step에 vertical line + label을 표시한다.
    마지막 boundary entry는 발화 끝 sentinel(label=None) — UI가 i번째 어절을
    boundaries[i].path_step ~ boundaries[i+1].path_step으로 추출 가능.

    Returns:
        {
          "learner_f0_zscore": list[float],  # path step별
          "native_f0_zscore":  list[float],
          "learner_time_at_step": list[float],  # 초 단위
          "eojeol_boundaries": list[{path_step: int, label: str|None}]
        }
    """
    l_mfcc = _cmvn(learner.mfcc)
    n_mfcc = _cmvn(native.mfcc)
    wp = _align_dtw(l_mfcc, n_mfcc)
    if wp is None or len(wp) == 0:
        return {
            "learner_f0_zscore": [],
            "native_f0_zscore": [],
            "learner_time_at_step": [],
            "eojeol_boundaries": [],
        }

    l_idx, n_idx = wp[:, 0], wp[:, 1]
    l_t = learner.times[l_idx]
    n_t = native.times[n_idx]
    l_f0 = f0_zscore_at_times(learner_f0, l_t)
    n_f0 = f0_zscore_at_times(native_f0, n_t)

    n_eo = min(len(eojeol_learner_spans), len(eojeol_labels))
    boundaries: list[dict] = []
    for i in range(n_eo):
        t0 = eojeol_learner_spans[i][0]
        step = int(np.searchsorted(l_t, t0))
        step = min(step, len(l_t) - 1)
        boundaries.append({"path_step": step, "label": eojeol_labels[i]})
    if n_eo > 0:
        boundaries.append({"path_step": len(l_t) - 1, "label": None})

    return {
        "learner_f0_zscore": [float(v) for v in l_f0],
        "native_f0_zscore": [float(v) for v in n_f0],
        "learner_time_at_step": [float(v) for v in l_t],
        "eojeol_boundaries": boundaries,
    }
