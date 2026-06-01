"""분절 축. 발화를 시간 구간(초)으로 자른다.

forced_alignment(phoneme_segments)를 아는 곳은 **여기뿐**.
WholeSegmenter는 아무 의존성도 없다 — 글로벌 DTW 렌즈가 쓴다.
공통 계약: native_spans()/learner_spans() → list[(t_start, t_end)], labels().
"""
from __future__ import annotations

from core.f0_extractor import F0Result
from core.syllable_utils import segments_to_syllable_boundaries

_HANGUL_LO, _HANGUL_HI = 0xAC00, 0xD7A3


def _end_time(f0: F0Result) -> float:
    if len(f0.times) == 0:
        return 0.0
    return float(f0.times[-1])


class WholeSegmenter:
    """발화 전체를 1구간. 의존성 0. blank 자를지 말지 질문 자체가 없음."""

    def __init__(self, native_f0: F0Result, learner_f0: F0Result):
        self._n = [(0.0, _end_time(native_f0))]
        self._l = [(0.0, _end_time(learner_f0))]

    def native_spans(self) -> list[tuple[float, float]]:
        return self._n

    def learner_spans(self) -> list[tuple[float, float]]:
        return self._l

    def labels(self) -> list[str]:
        return ["전체"]


class SyllableSegmenter:
    """음절 단위. forced_alignment의 phoneme_segments에 의존."""

    def __init__(
        self,
        native_segments: list[dict],
        learner_segments: list[dict],
        syllable_positions: list[str],
        syllable_labels: list[str] | None = None,
    ):
        self._n = segments_to_syllable_boundaries(native_segments, syllable_positions)
        self._l = segments_to_syllable_boundaries(learner_segments, syllable_positions)
        self._labels = syllable_labels or [""] * len(self._n)

    def native_spans(self) -> list[tuple[float, float]]:
        return self._n

    def learner_spans(self) -> list[tuple[float, float]]:
        return self._l

    def labels(self) -> list[str]:
        return self._labels[: len(self._n)]


class EojeolSegmenter:
    """어절 단위. 음절 경계를 어절별 음절 수로 묶는다. blank가 줄어든다."""

    def __init__(
        self,
        native_segments: list[dict],
        learner_segments: list[dict],
        syllable_positions: list[str],
        eojeol_text: str,
    ):
        n_syl = segments_to_syllable_boundaries(native_segments, syllable_positions)
        l_syl = segments_to_syllable_boundaries(learner_segments, syllable_positions)
        words = eojeol_text.split()
        counts = [
            sum(1 for ch in w if _HANGUL_LO <= ord(ch) <= _HANGUL_HI)
            for w in words
        ]
        self._n = _group(n_syl, counts)
        self._l = _group(l_syl, counts)
        self._labels = words[: len(self._n)]

    def native_spans(self) -> list[tuple[float, float]]:
        return self._n

    def learner_spans(self) -> list[tuple[float, float]]:
        return self._l

    def labels(self) -> list[str]:
        return self._labels


def _group(
    syl_spans: list[tuple[float, float]], counts: list[int]
) -> list[tuple[float, float]]:
    """음절 span들을 어절별 음절 수만큼 순차 묶음. 모자라면 남은 건 마지막에."""
    out: list[tuple[float, float]] = []
    i = 0
    for k, c in enumerate(counts):
        if i >= len(syl_spans):
            break
        last = k == len(counts) - 1
        end = len(syl_spans) if last else min(i + c, len(syl_spans))
        chunk = syl_spans[i:end]
        if chunk:
            out.append((chunk[0][0], chunk[-1][1]))
        i = end
    if i < len(syl_spans) and out:
        out[-1] = (out[-1][0], syl_spans[-1][1])
    return out
