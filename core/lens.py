"""렌즈 = Segmenter + Aligner를 끼워 PlotModel 1개를 만든다.

여기가 유일한 조합 지점. 새 렌즈는 (segmenter, aligner) 쌍을 넘기면 끝.
plotter는 결과 PlotModel만 받는다 (어떻게 잘리고 정렬됐는지 모름).
"""
from __future__ import annotations

import numpy as np

from core.comparator import SyllableComparison
from core.f0_extractor import F0Result
from core.features import f0_feature, slice_signal
from core.metrics import compute_metrics
from core.plot_model import PlotModel, Series, Span

NATIVE_COLOR = "#185FA5"
LEARNER_COLOR = "#993C1D"



def build_plot_model(
    native_f0: F0Result,
    learner_f0: F0Result,
    segmenter,
    aligner,
    title: str = "억양 비교",
    native_label: str = "원어민",
    learner_label: str = "학습자",
    pearson_threshold: float = 0.5,
    feature=f0_feature,
    y_axis_title: str = "음높이 (z-score)",
    y_range: tuple[float, float] | None = (-3.5, 3.0),
) -> PlotModel | None:
    nb = segmenter.native_spans()
    lb = segmenter.learner_spans()
    labels = segmenter.labels()
    if not nb or not lb:
        return None

    # feature(f0 / delta-f0)는 전체 contour에서 1회 계산 후 슬라이스
    # (어절 경계마다 미분 edge artifact가 생기지 않도록 whole-utterance 변환)
    n_feat = feature(native_f0)
    l_feat = feature(learner_f0)

    comparisons: list[SyllableComparison] = []
    seg_curves: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, ((n0, n1), (l0, l1)) in enumerate(zip(nb, lb)):
        n_f0, n_v, _ = slice_signal(native_f0.times, n_feat, native_f0.voiced_mask, n0, n1)
        l_f0, l_v, _ = slice_signal(learner_f0.times, l_feat, learner_f0.voiced_mask, l0, l1)
        _, nc, lc, jv = aligner.align(n_f0, n_v, l_f0, l_v)
        seg_curves.append((nc, lc))
        comparisons.append(SyllableComparison(
            syllable_idx=idx,
            native_f0=nc,
            learner_f0=lc,
            joint_voiced_mask=jv,
            native_duration=n1 - n0,
            learner_duration=l1 - l0,
        ))

    metrics = compute_metrics(comparisons)

    native_all = np.concatenate([nc for nc, _ in seg_curves])
    learner_all = np.concatenate([lc for _, lc in seg_curves])
    x = np.arange(len(native_all))

    spans: list[Span] = []
    offset = 0
    for idx, (nc, _) in enumerate(seg_curves):
        m = metrics[idx]
        ann = {"R": "N/A" if np.isnan(m.rmse) else f"{m.rmse:.2f}"}
        highlight = False
        if not np.isnan(m.pearson):
            ann["P"] = f"{m.pearson:.2f}"
            highlight = m.pearson < pearson_threshold
        label = labels[idx] if idx < len(labels) else ""
        spans.append(Span(
            start_idx=offset,
            end_idx=offset + len(nc),
            label=label,
            annotations=ann,
            highlight=highlight,
        ))
        offset += len(nc)

    return PlotModel(
        x=x,
        series=[
            Series(native_label, native_all, NATIVE_COLOR),
            Series(learner_label, learner_all, LEARNER_COLOR),
        ],
        spans=spans,
        title=title,
        x_axis_title="프레임 (정렬 후)",
        y_axis_title=y_axis_title,
        y_range=y_range,
        zero_line=True,
    )
