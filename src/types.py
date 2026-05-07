from __future__ import annotations

"""Shared dataclasses for the pronunciation evaluator."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhoneToken:
    symbol: str
    category: str
    syllable_position: str = "unknown"
    confidence: float | None = None


@dataclass
class IPASequence:
    raw_text: str
    normalized_text: str
    tokens: list[PhoneToken]

    @property
    def token_symbols(self) -> list[str]:
        return [token.symbol for token in self.tokens]


@dataclass
class PronunciationCandidate:
    pronunciation: str
    ipa: IPASequence
    notes: list[str] = field(default_factory=list)
    is_primary: bool = False


@dataclass
class PronunciationReference:
    original_text: str
    normalized_text: str
    representative_pronunciation: str
    representative_ipa: IPASequence
    candidates: list[PronunciationCandidate]


@dataclass
class AudioRecognitionResult:
    raw_text: str
    normalized_text: str
    tokens: list[PhoneToken]
    raw_label_text: str = ""
    raw_labels: list[str] = field(default_factory=list)
    logits: list[list[float]] | None = None
    frame_confidence: list[float] | None = None
    frame_timestamps: list[float] | None = None
    sampling_rate: int = 16000
    quality_report: AudioQualityReport | None = None


@dataclass
class AudioQualityReport:
    passed: bool
    duration_sec: float
    rms_db: float
    silence_ratio: float
    clipping_ratio: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class AlignmentSegment:
    token: str
    label: str
    start_time: float
    end_time: float
    frame_start: int
    frame_end: int
    confidence: float


@dataclass
class ForcedAlignmentResult:
    labels: list[str]
    token_symbols: list[str]
    segments: list[AlignmentSegment]
    total_log_prob: float
    normalized_log_prob: float
    avg_token_confidence: float
    coverage: float
    blank_ratio: float


@dataclass
class AlignmentConfidenceReport:
    passed: bool
    avg_token_confidence: float
    coverage: float
    normalized_log_prob: float
    message: str


@dataclass
class AlignmentStep:
    op: str
    ref_token: PhoneToken | None
    hyp_token: PhoneToken | None
    cost: float
    error_type: str
    detail: str
    ref_index: int | None = None
    hyp_index: int | None = None
    feature_penalties: dict[str, float] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    total_cost: float
    max_cost: float
    normalized_score: float
    ops: list[AlignmentStep]
    aligned_ref: list[str]
    aligned_hyp: list[str]
    segment_errors: list[str]
    feature_penalties: dict[str, float]
    selected_reference_candidate: PronunciationCandidate


@dataclass
class ScoreBreakdown:
    overall: float
    consonant: float
    vowel: float
    coda: float
    fluency_like: float
    raw_cost: float
    max_cost: float
    penalty_summary: list[str]


@dataclass
class PronunciationIssue:
    issue_type: str
    severity: str
    description: str
    tip: str
    ref_token: str | None
    hyp_token: str | None
    cost: float
    acceptable: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackReport:
    summary_level: str
    headline: str
    issues: list[PronunciationIssue]
    tips: list[str]
    debug_notes: list[str]


@dataclass
class EvaluationResult:
    evaluation_status: str
    status_message: str
    reference_text: str
    reference_pronunciation: str
    reference_ipa: IPASequence
    reference_candidates: list[PronunciationCandidate]
    selected_reference_candidate: PronunciationCandidate
    user_ipa_raw: str
    user_ipa_normalized: str
    user_tokens: list[PhoneToken]
    alignment_result: AlignmentResult | None
    score_breakdown: ScoreBreakdown | None
    feedback_report: FeedbackReport | None
    quality_report: AudioQualityReport | None = None
    forced_alignment_result: ForcedAlignmentResult | None = None
    alignment_confidence_report: AlignmentConfidenceReport | None = None
    debug: dict[str, Any] = field(default_factory=dict)
