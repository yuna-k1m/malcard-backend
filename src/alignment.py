from __future__ import annotations

"""Weighted alignment between reference and recognized IPA token sequences."""

from math import inf

from src.cost_model import deletion_cost, insertion_cost
from src.ipa_utils import build_ipa_sequence
from src.types import AlignmentResult, AlignmentStep, PhoneToken, PronunciationCandidate


def align_ipa_sequences(ref_tokens: list[PhoneToken], hyp_tokens: list[PhoneToken], cost_model_module, profile: str = "ru") -> AlignmentResult:
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[inf] * (n + 1) for _ in range(m + 1)]
    back: list[list[tuple[str, float, str, dict[str, float], str] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        cost, label = cost_model_module.deletion_cost(ref_tokens[i - 1], profile=profile)
        dp[i][0] = dp[i - 1][0] + cost
        back[i][0] = ("delete", cost, label, {}, f"{ref_tokens[i - 1].symbol} deleted")

    for j in range(1, n + 1):
        cost, label = cost_model_module.insertion_cost(hyp_tokens[j - 1], profile=profile)
        dp[0][j] = dp[0][j - 1] + cost
        back[0][j] = ("insert", cost, label, {}, f"{hyp_tokens[j - 1].symbol} inserted")

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost, sub_label, penalties, detail = cost_model_module.substitution_cost(ref_tokens[i - 1], hyp_tokens[j - 1], profile=profile)
            del_cost, del_label = cost_model_module.deletion_cost(ref_tokens[i - 1], profile=profile)
            ins_cost, ins_label = cost_model_module.insertion_cost(hyp_tokens[j - 1], profile=profile)
            candidates = [
                (dp[i - 1][j - 1] + sub_cost, ("match" if sub_cost == 0 else "substitute", sub_cost, sub_label, penalties, detail)),
                (dp[i - 1][j] + del_cost, ("delete", del_cost, del_label, {}, f"{ref_tokens[i - 1].symbol} deleted")),
                (dp[i][j - 1] + ins_cost, ("insert", ins_cost, ins_label, {}, f"{hyp_tokens[j - 1].symbol} inserted")),
            ]
            best_cost, best_back = min(candidates, key=lambda item: item[0])
            dp[i][j] = best_cost
            back[i][j] = best_back

    i, j = m, n
    steps: list[AlignmentStep] = []
    aligned_ref: list[str] = []
    aligned_hyp: list[str] = []
    feature_penalties: dict[str, float] = {}
    segment_errors: list[str] = []

    while i > 0 or j > 0:
        action = back[i][j]
        if action is None:
            break
        op, cost, label, penalties, detail = action
        ref_token = ref_tokens[i - 1] if op in {"match", "substitute", "delete"} and i > 0 else None
        hyp_token = hyp_tokens[j - 1] if op in {"match", "substitute", "insert"} and j > 0 else None
        steps.append(
            AlignmentStep(
                op=op,
                ref_token=ref_token,
                hyp_token=hyp_token,
                cost=cost,
                error_type=label,
                detail=detail,
                ref_index=i - 1 if ref_token is not None else None,
                hyp_index=j - 1 if hyp_token is not None else None,
                feature_penalties=penalties,
            )
        )
        for key, value in penalties.items():
            feature_penalties[key] = feature_penalties.get(key, 0.0) + value
        if cost > 0:
            segment_errors.append(detail)
        aligned_ref.append(ref_token.symbol if ref_token is not None else "∅")
        aligned_hyp.append(hyp_token.symbol if hyp_token is not None else "∅")
        if op in {"match", "substitute"}:
            i -= 1
            j -= 1
        elif op == "delete":
            i -= 1
        else:
            j -= 1

    steps.reverse()
    aligned_ref.reverse()
    aligned_hyp.reverse()

    max_cost = 0.0
    for token in ref_tokens:
        base, _ = deletion_cost(token, profile=profile)
        max_cost += base + 0.6
    for token in hyp_tokens:
        base, _ = insertion_cost(token, profile=profile)
        max_cost += base * 0.2

    normalized_score = max(0.0, min(100.0, 100.0 * (1.0 - (dp[m][n] / max(max_cost, 1e-6)))))

    placeholder = PronunciationCandidate(pronunciation="", ipa=build_ipa_sequence(""))
    return AlignmentResult(dp[m][n], max_cost, normalized_score, steps, aligned_ref, aligned_hyp, segment_errors, feature_penalties, placeholder)


def score_reference_candidates(reference_candidates: list[PronunciationCandidate], hyp_tokens: list[PhoneToken], cost_model_module, profile: str = "ru") -> AlignmentResult:
    if not reference_candidates:
        raise ValueError("정답 발음 후보가 없습니다.")
    best: AlignmentResult | None = None
    for candidate in reference_candidates:
        result = align_ipa_sequences(candidate.ipa.tokens, hyp_tokens, cost_model_module=cost_model_module, profile=profile)
        result.selected_reference_candidate = candidate
        if best is None or result.total_cost < best.total_cost:
            best = result
    assert best is not None
    return best
