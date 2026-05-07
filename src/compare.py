from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Dict, Any


@dataclass
class IPACompareResult:
    score_percent: float
    target_tokens: List[str]
    user_tokens: List[str]
    op_summary: Dict[str, int]
    details: List[Dict[str, Any]]


def compare_ipa_tokens(target_tokens: List[str], user_tokens: List[str]) -> IPACompareResult:
    matcher = SequenceMatcher(a=target_tokens, b=user_tokens)
    ratio = matcher.ratio() * 100.0

    details: List[Dict[str, Any]] = []
    op_summary = {"equal": 0, "replace": 0, "delete": 0, "insert": 0}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        tgt = target_tokens[i1:i2]
        usr = user_tokens[j1:j2]
        op_summary[tag] += max(i2 - i1, j2 - j1)

        details.append(
            {
                "tag": tag,
                "target": tgt,
                "user": usr,
            }
        )

    return IPACompareResult(
        score_percent=round(ratio, 2),
        target_tokens=target_tokens,
        user_tokens=user_tokens,
        op_summary=op_summary,
        details=details,
    )


def render_diff_lines(details: List[Dict[str, Any]]) -> str:
    lines = []
    for item in details:
        tag = item["tag"]
        target = " ".join(item["target"]) if item["target"] else "∅"
        user = " ".join(item["user"]) if item["user"] else "∅"
        lines.append(f"[{tag}] target: {target} | user: {user}")
    return "\n".join(lines)