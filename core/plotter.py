"""HTML(plotly) 시각화. Plotter는 멍청이 — PlotModel만 받아 그린다.

`figure_from_model`은 정렬/분절 방식을 전혀 모른다. comparator·forced_alignment·
f0_extractor를 import하지 않는다. 새 렌즈는 PlotModel을 만드는 어댑터만 추가하면
되고, 이 파일은 건드리지 않는다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.plot_model import PlotModel, Series, Span

if TYPE_CHECKING:
    from core.comparator import SyllableComparison
    from core.f0_extractor import F0Result
    from core.metrics import SyllableMetrics

NATIVE_COLOR = "#185FA5"
LEARNER_COLOR = "#993C1D"


def figure_from_model(model: PlotModel) -> go.Figure:
    """PlotModel → plotly Figure. 이게 유일한 그리기 진입점(멍청이)."""
    fig = go.Figure()

    if model.zero_line:
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)

    for s in model.series:
        fig.add_trace(go.Scatter(
            x=model.x,
            y=s.y,
            mode="lines",
            name=s.name,
            line=dict(color=s.color, width=2.5),
            hovertemplate=f"{s.name}<br>frame=%{{x}}<br>y=%{{y:.3f}}<extra></extra>",
        ))

    y_lo, y_hi = model.y_range if model.y_range else (None, None)
    for sp in model.spans:
        fig.add_vline(x=sp.start_idx, line_dash="dash", line_color="gray", opacity=0.4)
        mid = (sp.start_idx + sp.end_idx) / 2
        color = "red" if sp.highlight else "gray"

        if sp.label:
            fig.add_annotation(
                x=mid, y=(y_lo if y_lo is not None else 0),
                text=sp.label, showarrow=False,
                yshift=-22, font=dict(size=12, color=color),
            )
        if sp.annotations:
            text = "<br>".join(f"{k}:{v}" for k, v in sp.annotations.items())
            fig.add_annotation(
                x=mid, y=(y_hi if y_hi is not None else 0),
                text=text, showarrow=False,
                yshift=-10, font=dict(size=11, color=color),
                align="center",
            )

    fig.update_layout(
        title=model.title,
        xaxis_title=model.x_axis_title,
        yaxis_title=model.y_axis_title,
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if model.y_range:
        fig.update_yaxes(range=list(model.y_range))
    return fig


class ComparisonPlotter:
    """기존 SyllableComparison/SyllableMetrics → PlotModel 어댑터.

    판정/정렬 로직은 모른다. comparisons를 받아 그릴 수 있는 모델로만 바꾼다.
    """

    def __init__(
        self,
        native_label: str = "원어민",
        learner_label: str = "학습자",
        threshold: float = 1,
    ):
        self.native_label = native_label
        self.learner_label = learner_label
        self.threshold = threshold

    def build_model(
        self,
        comparisons: list[SyllableComparison],
        metrics: list[SyllableMetrics],
        title: str = "억양 비교",
        syllable_labels: list[str] | None = None,
    ) -> PlotModel:
        n_frames = len(comparisons[0].native_f0)
        native_all = np.concatenate([c.native_f0 for c in comparisons])
        learner_all = np.concatenate([c.learner_f0 for c in comparisons])
        x = np.arange(len(native_all))

        spans: list[Span] = []
        for i, m in enumerate(metrics):
            ann: dict[str, str] = {}
            ann["R"] = "N/A" if np.isnan(m.rmse) else f"{m.rmse:.2f}"
            highlight = False
            if not np.isnan(m.pearson):
                ann["P"] = f"{m.pearson:.2f}"
                highlight = m.pearson < 0.5
            label = ""
            if syllable_labels and i < len(syllable_labels):
                label = syllable_labels[i]
            spans.append(Span(
                start_idx=i * n_frames,
                end_idx=(i + 1) * n_frames,
                label=label,
                annotations=ann,
                highlight=highlight,
            ))

        return PlotModel(
            x=x,
            series=[
                Series(self.native_label, native_all, NATIVE_COLOR),
                Series(self.learner_label, learner_all, LEARNER_COLOR),
            ],
            spans=spans,
            title=title,
            x_axis_title="리샘플 프레임",
            y_axis_title="음높이 (z-score)",
            y_range=(-3.5, 3.0),
            zero_line=True,
        )

    def plot(
        self,
        comparisons: list[SyllableComparison],
        metrics: list[SyllableMetrics],
        title: str = "억양 비교",
        syllable_labels: list[str] | None = None,
        save_path: str | None = None,
    ) -> go.Figure | None:
        if not comparisons:
            return None
        fig = figure_from_model(
            self.build_model(comparisons, metrics, title, syllable_labels)
        )
        if save_path:
            fig.write_html(save_path)
        return fig

    def plot_raw_f0(
        self,
        native: F0Result,
        learner: F0Result,
        title: str = "Normalized Pitch 비교 (Hz)",
        save_path: str | None = None,
    ) -> go.Figure:
        """DTW 정렬 없이 각 발화의 실제 시간축 F0(Hz)를 위아래로 비교.

        f0 추출은 호출부 책임 — plotter는 F0Result만 받는다.
        """
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            subplot_titles=(self.native_label, self.learner_label),
        )
        for row, f0r, color in (
            (1, native, NATIVE_COLOR),
            (2, learner, LEARNER_COLOR),
        ):
            fig.add_trace(
                go.Scatter(
                    x=f0r.times, y=f0r.f0_raw, mode="lines",
                    line=dict(color=color, width=2),
                    hovertemplate="t=%{x:.3f}s<br>F0=%{y:.1f}Hz<extra></extra>",
                ),
                row=row, col=1,
            )
            fig.update_yaxes(title_text="F0 (Hz)", row=row, col=1)
            fig.update_xaxes(title_text="시간 (초)", row=row, col=1)

        fig.update_layout(
            title=title, template="plotly_white",
            height=600, showlegend=False,
        )
        if save_path:
            fig.write_html(save_path)
        return fig
