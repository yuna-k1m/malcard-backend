"""Plotter seam의 데이터 계약.

이 모듈은 numpy 외 의존성이 없다. Aligner/Segmenter가 무엇을 하든,
plotter는 PlotModel 하나만 받아서 그린다. 렌즈(raw/음절/어절/DTW)를
바꿔도 plotter는 한 줄도 바뀌지 않는다.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Series:
    name: str
    y: np.ndarray
    color: str


@dataclass
class Span:
    """x축의 [start_idx, end_idx) 한 구간. 어떻게 잘렸는지 plotter는 모른다."""

    start_idx: int
    end_idx: int
    label: str = ""
    annotations: dict[str, str] = field(default_factory=dict)
    highlight: bool = False  # 이탈 구간 강조 (빨간색)


@dataclass
class PlotModel:
    x: np.ndarray
    series: list[Series]
    spans: list[Span] = field(default_factory=list)
    title: str = ""
    x_axis_title: str = ""
    y_axis_title: str = ""
    y_range: tuple[float, float] | None = None
    zero_line: bool = False
