from __future__ import annotations

from typing import List, Dict, Any


def build_feedback(details: List[Dict[str, Any]], score_percent: float) -> str:
    lines: List[str] = []

    if score_percent >= 90:
        lines.append("전체적으로 목표 IPA와 매우 가깝습니다.")
    elif score_percent >= 75:
        lines.append("전체적으로 비슷하지만 몇몇 음소 차이가 있습니다.")
    elif score_percent >= 55:
        lines.append("중간 정도 일치합니다. 특정 구간의 음소 차이가 눈에 띕니다.")
    else:
        lines.append("목표 IPA와 차이가 큽니다. 천천히 또렷하게 다시 발음해보는 것이 좋습니다.")

    replace_examples = []
    delete_examples = []
    insert_examples = []

    for item in details:
        tag = item["tag"]
        tgt = " ".join(item["target"]) if item["target"] else "∅"
        usr = " ".join(item["user"]) if item["user"] else "∅"

        if tag == "replace" and len(replace_examples) < 3:
            replace_examples.append(f"{tgt} → {usr}")
        elif tag == "delete" and len(delete_examples) < 3:
            delete_examples.append(f"{tgt} 누락")
        elif tag == "insert" and len(insert_examples) < 3:
            insert_examples.append(f"{usr} 추가")

    if replace_examples:
        lines.append("치환 구간: " + ", ".join(replace_examples))
    if delete_examples:
        lines.append("누락 구간: " + ", ".join(delete_examples))
    if insert_examples:
        lines.append("추가 구간: " + ", ".join(insert_examples))

    lines.append("주의: 이 프로토타입은 직접 음성에서 phone/IPA 시퀀스를 추정하는 최소 버전이라 표기 체계 차이와 모델 오차가 일부 남을 수 있습니다.")
    return "\n".join(lines)