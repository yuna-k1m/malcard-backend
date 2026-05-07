from __future__ import annotations

"""Scoring helpers built on top of weighted alignment."""

from src.types import AlignmentResult, ScoreBreakdown


def _score_from_cost(cost: float, max_cost: float) -> float:
    if max_cost <= 0:
        return 100.0
    return max(0.0, min(100.0, 100.0 * (1.0 - cost / max_cost)))


def build_score_breakdown(alignment_result: AlignmentResult) -> ScoreBreakdown:
    consonant_cost = 0.0
    vowel_cost = 0.0
    coda_cost = 0.0
    consonant_max = 0.0
    vowel_max = 0.0
    coda_max = 0.0
    penalty_summary: list[tuple[float, str]] = []

    for step in alignment_result.ops:
        ref = step.ref_token
        hyp = step.hyp_token
        ref_category = ref.category if ref is not None else (hyp.category if hyp is not None else "other")
        weight = 1.2 if ref_category == "consonant" else 1.0
        if ref is not None and ref.syllable_position == "coda":
            weight = 1.4

        if ref_category == "vowel":
            vowel_max += weight
            vowel_cost += step.cost
        elif ref is not None and ref.syllable_position == "coda":
            coda_max += weight
            coda_cost += step.cost
            consonant_max += 0.3
            consonant_cost += step.cost * 0.3
        else:
            consonant_max += weight
            consonant_cost += step.cost

        if step.cost > 0:
            penalty_summary.append((step.cost, f"{step.error_type}: {step.detail} (-{step.cost:.2f})"))

    overall = _score_from_cost(alignment_result.total_cost, alignment_result.max_cost)
    consonant = _score_from_cost(consonant_cost, max(consonant_max, 1.0))
    vowel = _score_from_cost(vowel_cost, max(vowel_max, 1.0))
    coda = _score_from_cost(coda_cost, max(coda_max, 1.0))
    fluency_like = max(0.0, min(100.0, overall - max(0.0, len([step for step in alignment_result.ops if step.op == "insert"]) - 1) * 4.0))
    top_penalties = [text for _, text in sorted(penalty_summary, key=lambda item: item[0], reverse=True)[:5]]

    return ScoreBreakdown(overall, consonant, vowel, coda, fluency_like, alignment_result.total_cost, alignment_result.max_cost, top_penalties)
