from __future__ import annotations

"""Interpret alignment steps as learner-facing pronunciation issues."""

from src.types import AlignmentResult, PronunciationIssue


def classify_alignment_errors(alignment_result: AlignmentResult, profile: str = "ru") -> list[PronunciationIssue]:
    issues: list[PronunciationIssue] = []
    for step in alignment_result.ops:
        if step.cost <= 0:
            continue
        ref_symbol = step.ref_token.symbol if step.ref_token is not None else None
        hyp_symbol = step.hyp_token.symbol if step.hyp_token is not None else None
        acceptable = step.cost <= 0.2 and step.error_type in {"coda neutralization", "feature-similar substitution"}
        severity = "high" if step.cost >= 0.8 else "medium" if step.cost >= 0.4 else "low"

        if step.error_type == "laryngeal contrast confusion":
            description = "기식성이나 경음 대조가 약하게 실현되었습니다."
            tip = "숨을 더 실어 내거나 성대 긴장을 조금 더 주어 ㅋ/ㅌ/ㅍ, ㄲ/ㄸ/ㅃ 계열 차이를 분명하게 내보세요."
            issue_type = "aspiration_or_tense_confusion"
        elif step.error_type == "vowel confusion":
            description = "모음 높이 또는 입모양 차이가 기준 발음과 다르게 들렸습니다."
            tip = "입 벌림과 혀 위치를 조금 더 크게 구분해 보세요. 특히 ㅓ/ㅗ, ㅡ/ㅜ, ㅐ/ㅔ 대비를 의식하면 도움이 됩니다."
            issue_type = "vowel_confusion"
        elif step.error_type == "liquid realization issue":
            description = "ㄹ 계열 소리가 플랩/유음 사이에서 흔들렸습니다."
            tip = "혀끝을 윗잇몸에 짧게 한 번 닿게 하면 초성 ㄹ에, 조금 더 길게 닿게 하면 종성 ㄹ에 가깝습니다."
            issue_type = "liquid_realization_issue"
        elif step.error_type == "coda deletion":
            description = "받침이 약하게 실현되어 거의 들리지 않았습니다."
            tip = "음절 끝을 길게 빼지 말고, 짧게 막아 주는 느낌으로 받침을 마무리해 보세요."
            issue_type = "coda_deletion"
        elif step.error_type == "epenthetic vowel insertion":
            description = "기준 발음에는 없는 모음이 사이에 덧붙었습니다."
            tip = "자음 사이를 연결할 때 불필요한 모음을 넣지 않도록 리듬을 더 짧게 가져가 보세요."
            issue_type = "epenthetic_vowel_insertion"
        elif step.error_type == "consonant insertion":
            description = "기준 발음에는 없는 자음이 추가되어 들렸습니다."
            tip = "강하게 끊는 구간에서 불필요한 자음이 섞이지 않게 호흡을 조금 더 매끄럽게 이어 보세요."
            issue_type = "consonant_insertion"
        elif step.error_type == "consonant deletion":
            description = "기준 자음이 약화되거나 빠져서 들렸습니다."
            tip = "문장 속 자음을 흘리지 말고, 시작점에서 짧게 또렷하게 터뜨려 보세요."
            issue_type = "consonant_deletion"
        else:
            description = "기준 발음과 다른 구간이 확인되었습니다."
            tip = "해당 구간을 천천히 반복해 기준 발음과의 차이를 줄여 보세요."
            issue_type = step.error_type.replace(" ", "_")

        if profile == "ru" and issue_type in {"aspiration_or_tense_confusion", "liquid_realization_issue", "vowel_confusion"}:
            description += " 러시아어권 학습자에게 자주 보이는 유형이라 대조를 더 크게 내면 도움이 됩니다."

        issues.append(PronunciationIssue(issue_type, severity, description, tip, ref_symbol, hyp_symbol, step.cost, acceptable, {"detail": step.detail, "feature_penalties": step.feature_penalties}))

    issues.sort(key=lambda issue: issue.cost, reverse=True)
    return issues
