"""Records → deterministic 자연어 feedback (친근체·권유형, 어린아이 톤).

설계 (grill 합의):
- LLM 호출 없음. record와 NL 1:1 mapping.
- 5 rule_label별 template (hardcode — NL 표현은 reactive로 micro 조정).
- pitch shape rule(rising/falling)은 windows의 learner_time_ratio center로
  bucket 분류 (초반/중반/후반/어절 전반). syllable_hint는 evidence에만 남김.
- severity는 부사 swap (minor='조금', '조금만 더' / major='너무', '확실히 더').
- records=[]면 top-level praise field.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from core.record import Record

_PRAISE = "잘 발화했어요!"

# severity → 부사 (3종 — 문맥별로 골라 씀)
_DEGREE = {"minor": "조금", "major": "너무"}            # 상태 표현 ("조금 올라갔어요")
_MORE = {"minor": "조금만 더", "major": "확실히 더"}    # pitch shape 교정 ("조금만 더 평평하게")
_INTENSITY = {"minor": "조금만", "major": "확실히"}     # 단순 교정 ("조금만 낮게")


def _classify_bucket(windows: list[dict]) -> str:
    """windows의 learner_time_ratio center bucket counts → 최빈 bucket.

    bucket: <0.3 초반 / 0.3~0.7 중반 / >=0.7 후반.
    최빈이 단독이면 그 이름, tie면 '어절 전반'.
    """
    counts = {"초반": 0, "중반": 0, "후반": 0}
    for w in windows:
        a, b = w["learner_time_ratio"]
        center = (a + b) / 2
        if center < 0.3:
            counts["초반"] += 1
        elif center < 0.7:
            counts["중반"] += 1
        else:
            counts["후반"] += 1
    max_count = max(counts.values())
    leaders = [b for b, c in counts.items() if c == max_count]
    return leaders[0] if len(leaders) == 1 else "어절 전반"


def _where_phrase(bucket: str) -> str:
    """bucket → 자연스러운 조사 부착."""
    return f"어절의 {bucket}에서" if bucket != "어절 전반" else "어절 전반에서"


def _format_pitch_rising_excess(record: Record) -> str:
    eo = record.evidence_metrics["eojeol_label"]
    bucket = _classify_bucket(record.evidence_metrics["windows"])
    return (
        f"'{eo}' {_where_phrase(bucket)} 음이 {_DEGREE[record.severity]} 올라갔어요. "
        f"{_MORE[record.severity]} 평평하게 말해봐요."
    )


def _format_pitch_falling_excess(record: Record) -> str:
    eo = record.evidence_metrics["eojeol_label"]
    bucket = _classify_bucket(record.evidence_metrics["windows"])
    return (
        f"'{eo}' {_where_phrase(bucket)} 음이 {_DEGREE[record.severity]} 떨어졌어요. "
        f"{_MORE[record.severity]} 평평하게 말해봐요."
    )


def _format_pitch_offset(record: Record) -> str:
    eo = record.evidence_metrics["eojeol_label"]
    z_diff = record.evidence_metrics["z_diff"]
    if z_diff > 0:
        current, action = "높", "낮게"
    else:
        current, action = "낮", "높게"
    return (
        f"'{eo}' 어절 전체 음높이가 {_DEGREE[record.severity]} {current}아요. "
        f"{_INTENSITY[record.severity]} {action} 말해봐요."
    )


def _format_syllable_elongation(record: Record) -> str:
    eo = record.evidence_metrics["eojeol_label"]
    syl = record.evidence_metrics["syllable_label"]
    return (
        f"'{eo}' 어절의 '{syl}' 음절을 {_DEGREE[record.severity]} 길게 발음했어요. "
        f"짧게 말해봐요."
    )


def _format_eojeol_slow(record: Record) -> str:
    eo = record.evidence_metrics["eojeol_label"]
    ratio = record.evidence_metrics["duration_ratio"]
    degree = _DEGREE[record.severity]
    if ratio > 1.0:
        return f"'{eo}' 어절 전체를 {degree} 천천히 발화했어요. 빠르게 말해봐요."
    return f"'{eo}' 어절 전체를 {degree} 빠르게 발화했어요. 천천히 말해봐요."


_FORMATTERS = {
    "pitch_rising_excess": _format_pitch_rising_excess,
    "pitch_falling_excess": _format_pitch_falling_excess,
    "pitch_offset": _format_pitch_offset,
    "syllable_elongation": _format_syllable_elongation,
    "eojeol_slow": _format_eojeol_slow,
}


def generate_feedback(record: Record) -> str:
    """rule_label switch → NL 한 문장."""
    return _FORMATTERS[record.rule_label](record)


def build_payload(records: list[Record], reference_text: str) -> dict[str, Any]:
    """records.json payload: 각 record에 feedback_text 추가, records=[]면 praise field."""
    rec_dicts = []
    for r in records:
        d = asdict(r)
        d["feedback_text"] = generate_feedback(r)
        rec_dicts.append(d)

    payload: dict[str, Any] = {
        "reference_text": reference_text,
        "records": rec_dicts,
    }
    if not records:
        payload["summary_when_no_outlier"] = _PRAISE
    return payload
