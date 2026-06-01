"""Evidence record schema — lens-rule paradigm의 코드↔LLM 데이터 계약.

코드(lens-rule trigger)가 outlier 어절을 식별·라벨링해 채운 뒤 LLM에 넘기면,
LLM은 record list만 받아 자연어 feedback을 합성한다.

설계 원칙(B Sieved + labeled):
- 코드가 outlier 판정·rule_label·severity·근거 metric까지 deterministic하게 결정
- LLM은 NL 합성·톤 조절·우선순위 처리만 (수치 비교는 LLM 약점)
- evidence_metrics는 rule별 dict — rule_label과 키 이름이 의미 전달 (TypedDict
  Union 대신 dict[str, Any]). 새 rule 추가 시 자기 dict 구조를 자기가 정의.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RuleLabel = Literal[
    "pitch_rising_excess",
    "pitch_falling_excess",
    "pitch_offset",
    "syllable_elongation",
    "eojeol_slow",
]

Severity = Literal["minor", "major"]


@dataclass
class Record:
    eojeol_idx: int
    rule_label: RuleLabel
    severity: Severity
    trigger_lens: str
    evidence_metrics: dict[str, Any] = field(default_factory=dict)
    syllable_hint: str | None = None

    def to_llm_dict(self) -> dict[str, Any]:
        """LLM prompt 직렬화 — trigger_lens(내부 디버깅용) 제외, null hint 제거."""
        d = asdict(self)
        d.pop("trigger_lens")
        if d["syllable_hint"] is None:
            d.pop("syllable_hint")
        return d


def sort_by_severity(records: list[Record]) -> list[Record]:
    """major 먼저. 같은 등급은 입력 순서 유지(안정 정렬)."""
    order = {"major": 0, "minor": 1}
    return sorted(records, key=lambda r: order[r.severity])
