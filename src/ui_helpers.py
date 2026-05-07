from __future__ import annotations

"""Helpers for presenting evaluation results in Streamlit."""

from src.types import AlignmentResult, FeedbackReport


def alignment_rows(alignment_result: AlignmentResult) -> list[dict[str, str | float | int | None]]:
    rows: list[dict[str, str | float | int | None]] = []
    for index, step in enumerate(alignment_result.ops, start=1):
        rows.append(
            {
                "순번": index,
                "기준": step.ref_token.symbol if step.ref_token is not None else "∅",
                "사용자": step.hyp_token.symbol if step.hyp_token is not None else "∅",
                "연산": step.op,
                "오류 유형": step.error_type,
                "비용": round(step.cost, 3),
                "설명": step.detail,
            }
        )
    return rows


def short_evaluation_line(feedback_report: FeedbackReport) -> str:
    return feedback_report.headline
