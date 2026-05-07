from __future__ import annotations

"""Build learner-facing Korean feedback from alignment issues."""

from src.types import FeedbackReport, PronunciationIssue, ScoreBreakdown


def issue_to_feedback(issue: PronunciationIssue, profile: str = "ru") -> str:
    token_text = ""
    if issue.ref_token or issue.hyp_token:
        token_text = f" 기준 {issue.ref_token or '∅'} / 사용자 {issue.hyp_token or '∅'}."
    return f"{issue.description}{token_text} {issue.tip}"


def build_feedback_report(issues: list[PronunciationIssue], score_breakdown: ScoreBreakdown, profile: str = "ru") -> FeedbackReport:
    visible_issues = [issue for issue in issues if not issue.acceptable][:5]
    if score_breakdown.overall >= 90:
        summary_level = "excellent"
        headline = "전체적으로 기준 발음과 매우 가깝습니다."
    elif score_breakdown.overall >= 75:
        summary_level = "good"
        headline = "전반적으로 안정적이지만 몇몇 대비를 더 분명하게 하면 좋습니다."
    elif score_breakdown.overall >= 55:
        summary_level = "developing"
        headline = "핵심 발음은 전달되지만 일부 구간에서 대조가 약해졌습니다."
    else:
        summary_level = "needs-work"
        headline = "기준 발음과 차이가 커서 구간별 교정이 필요합니다."

    tips: list[str] = []
    for issue in visible_issues:
        text = issue_to_feedback(issue, profile=profile)
        if text not in tips:
            tips.append(text)

    if not tips:
        tips.append("허용 가능한 발음 변이를 제외하면 큰 문제는 보이지 않았습니다. 현재 리듬을 유지하면서 또렷함만 조금 더 높여 보세요.")

    debug_notes = [
        f"overall={score_breakdown.overall:.2f}",
        f"consonant={score_breakdown.consonant:.2f}",
        f"vowel={score_breakdown.vowel:.2f}",
        f"coda={score_breakdown.coda:.2f}",
    ] + score_breakdown.penalty_summary

    return FeedbackReport(summary_level, headline, visible_issues, tips, debug_notes)
