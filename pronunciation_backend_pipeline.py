from __future__ import annotations

"""Backend-friendly pronunciation evaluation pipeline.

This module has no Streamlit dependency. Backend/API code can call
`evaluate_pronunciation_file(audio_path, reference_text)` and receive the two
handoff payloads needed by the feedback LLM and prosody analyzer.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any


from src import cost_model
from src.alignment import score_reference_candidates
from src.audio_to_ipa import AudioToIPARecognizer
from src.backdata_export import build_evaluation_backdata, save_evaluation_bundle
from src.confidence_calibration import load_confidence_calibration
from src.error_analysis import classify_alignment_errors
from src.feedback_report import build_feedback_report
from src.forced_alignment import force_align_candidate
from src.quality import (
    assess_alignment_confidence,
    decide_prosody_alignment_usage,
    summarize_alignment_timing,
)
from src.recognition import recognize_audio
from src.reference_builder import text_to_pronunciation
from src.scoring import build_score_breakdown
from src.types import (
    AlignmentConfidenceReport,
    AlignmentResult,
    AudioRecognitionResult,
    EvaluationResult,
    FeedbackReport,
    ForcedAlignmentResult,
    IPASequence,
    PronunciationCandidate,
    PronunciationReference,
    ScoreBreakdown,
)


COARSE_SIMILARITY_THRESHOLD = 75.0
TAIL_RETRY_MAX_TRIM = 2


_ALIGNMENT_QUALITY_RANK = {
    "discardable": 0,
    "usable_with_warnings": 1,
    "minor_warnings": 2,
    "clean": 3,
}


@dataclass
class PipelineContext:
    """Mutable state passed through the backend pipeline stages."""

    audio_path: Path
    reference_text: str
    profile: str
    coarse_similarity_threshold: float
    recognizer: AudioToIPARecognizer
    reference: PronunciationReference | None = None
    recognition: AudioRecognitionResult | None = None
    selected_candidate: PronunciationCandidate | None = None
    coarse_alignment: AlignmentResult | None = None
    forced_alignment: ForcedAlignmentResult | None = None
    alignment_confidence: AlignmentConfidenceReport | None = None
    score_breakdown: ScoreBreakdown | None = None
    feedback_report: FeedbackReport | None = None
    alignment_notes: list[str] = field(default_factory=list)
    acceptance_notes: list[str] = field(default_factory=list)
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    alignment_blank_id: int | None = None
    alignment_calibration_stats: dict[str, Any] | None = None


@dataclass
class _AlignmentAttempt:
    name: str
    result: ForcedAlignmentResult
    confidence: AlignmentConfidenceReport
    timing_summary: dict[str, Any]
    trim_count: int = 0


@lru_cache(maxsize=1)
def get_default_recognizer() -> AudioToIPARecognizer:
    """Load the heavy speech model once per backend process."""

    return AudioToIPARecognizer()


def build_reference_stage(ctx: PipelineContext) -> None:
    """Build pronunciation candidates and IPA references from the answer text."""

    ctx.reference = text_to_pronunciation(ctx.reference_text)
    ctx.selected_candidate = ctx.reference.candidates[0]


def recognize_audio_stage(ctx: PipelineContext) -> None:
    """Run the phone recognizer and keep logits/timestamps for alignment."""

    ctx.recognition = recognize_audio(ctx.recognizer, ctx.audio_path)


def audio_quality_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Record audio-quality failures while keeping the evaluation accepted."""

    assert ctx.reference is not None and ctx.recognition is not None and ctx.selected_candidate is not None
    report = ctx.recognition.quality_report
    if report is None or report.passed:
        return None

    ctx.acceptance_notes.append(
        "Audio quality gate failed but evaluation continued under soft-accept policy."
    )
    return None


def recognition_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Record empty recognition output while keeping the evaluation accepted."""

    assert ctx.reference is not None and ctx.recognition is not None
    if ctx.recognition.tokens:
        return None

    ctx.acceptance_notes.append(
        "Recognition produced no stable phone tokens but evaluation continued under soft-accept policy."
    )
    return None


def coarse_token_alignment_stage(ctx: PipelineContext) -> None:
    """Run weighted token-level IPA alignment against all reference candidates."""

    assert ctx.reference is not None and ctx.recognition is not None
    ctx.coarse_alignment = score_reference_candidates(
        ctx.reference.candidates,
        ctx.recognition.tokens,
        cost_model_module=cost_model,
        profile=ctx.profile,
    )
    ctx.selected_candidate = ctx.coarse_alignment.selected_reference_candidate


def coarse_token_alignment_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Record coarse mismatch while keeping the evaluation accepted."""

    assert ctx.coarse_alignment is not None and ctx.recognition is not None
    if ctx.coarse_alignment.normalized_score >= ctx.coarse_similarity_threshold:
        return None

    ctx.acceptance_notes.append(
        "Raw coarse IPA similarity gate failed; forced alignment can recover the coarse gate "
        "when reference-constrained alignment confidence passes. "
        f"coarse_similarity={ctx.coarse_alignment.normalized_score:.3f}, "
        f"threshold={ctx.coarse_similarity_threshold:.3f}"
    )
    return None


def _unstable_tail_trim_count(result: ForcedAlignmentResult) -> int:
    segments = result.segments
    if len(segments) < 3:
        return 0

    timing_summary = summarize_alignment_timing(result)
    gaps = [
        max(0.0, segments[index].start_time - segments[index - 1].end_time)
        for index in range(1, len(segments))
    ]
    final_confidences = [segment.confidence for segment in segments[-2:]]

    if len(final_confidences) == 2 and all(confidence < 0.05 for confidence in final_confidences):
        return 2

    if gaps and gaps[-1] > 0.30:
        return 1

    if len(gaps) >= 2 and gaps[-2] > 0.30:
        return 2

    final_duration = max(0.0, segments[-1].end_time - segments[-1].start_time)
    median_duration = timing_summary["median_duration"]
    final_duration_is_suspicious = (
        median_duration > 0.0
        and final_duration > max(0.75, median_duration * 8.0)
    )
    if (
        segments[-1].confidence < 0.005
        and (timing_summary["max_tail_gap"] > 0.30 or final_duration_is_suspicious)
    ):
        return 1

    return 0


def _trim_candidate_tail(candidate: PronunciationCandidate, trim_count: int) -> PronunciationCandidate:
    kept_tokens = candidate.ipa.tokens[:-trim_count]
    trimmed_tokens = candidate.ipa.tokens[-trim_count:]
    kept_text = " ".join(token.symbol for token in kept_tokens)
    trimmed_text = " ".join(token.symbol for token in trimmed_tokens)
    notes = [
        *candidate.notes,
        f"forced_alignment_tail_retry_excluded={trimmed_text}",
    ]
    return PronunciationCandidate(
        pronunciation=candidate.pronunciation,
        ipa=IPASequence(
            raw_text=kept_text,
            normalized_text=kept_text,
            tokens=kept_tokens,
        ),
        notes=notes,
        is_primary=candidate.is_primary,
    )


def _tail_region_max_gap(result: ForcedAlignmentResult, trim_count: int) -> float:
    segments = result.segments
    if len(segments) < 2:
        return 0.0
    gaps = [
        max(0.0, segments[index].start_time - segments[index - 1].end_time)
        for index in range(1, len(segments))
    ]
    return max(gaps[-trim_count:], default=0.0)


def _large_tail_trim_allowed(full_attempt: _AlignmentAttempt, trim_count: int) -> bool:
    segments = full_attempt.result.segments
    return 0 < trim_count <= TAIL_RETRY_MAX_TRIM and len(segments) > trim_count


def _tail_retry_trim_counts(candidate: PronunciationCandidate, full_attempt: _AlignmentAttempt) -> list[int]:
    max_trim = min(TAIL_RETRY_MAX_TRIM, max(0, len(candidate.ipa.tokens) - 1))
    return [
        trim_count
        for trim_count in range(1, max_trim + 1)
        if _large_tail_trim_allowed(full_attempt, trim_count)
    ]


def _get_alignment_vocab(recognizer: AudioToIPARecognizer) -> tuple[dict[str, int], int]:
    if hasattr(recognizer, "get_alignment_vocab"):
        return recognizer.get_alignment_vocab()
    tokenizer = getattr(recognizer.processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Recognizer processor does not expose a tokenizer for forced alignment.")
    return tokenizer.get_vocab(), tokenizer.pad_token_id


def _alignment_evidence(ctx: PipelineContext) -> dict[str, Any]:
    if ctx.recognition is None:
        return {
            "frame_confidence": None,
            "logits": None,
            "blank_id": ctx.alignment_blank_id,
            "calibration_stats": ctx.alignment_calibration_stats,
        }
    return {
        "frame_confidence": ctx.recognition.frame_confidence,
        "logits": ctx.recognition.logits,
        "blank_id": ctx.alignment_blank_id,
        "calibration_stats": ctx.alignment_calibration_stats,
    }


def _make_alignment_attempt(
    name: str,
    result: ForcedAlignmentResult,
    *,
    trim_count: int = 0,
    evidence: dict[str, Any] | None = None,
) -> _AlignmentAttempt:
    evidence = evidence or {}
    confidence = assess_alignment_confidence(result, **evidence)
    timing_summary = summarize_alignment_timing(result, **evidence)
    return _AlignmentAttempt(
        name=name,
        result=result,
        confidence=confidence,
        timing_summary=timing_summary,
        trim_count=trim_count,
    )


def _alignment_attempt_rank(attempt: _AlignmentAttempt) -> tuple:
    timing = attempt.timing_summary
    return (
        int(attempt.confidence.passed),
        _ALIGNMENT_QUALITY_RANK.get(timing.get("quality_level"), 0),
        float(attempt.result.coverage),
        -int(timing.get("failure_gap_count") or 0),
        -int(timing.get("warning_gap_count") or 0),
        -int(timing.get("large_gap_count") or 0),
        -float(timing.get("effective_very_low_confidence_ratio", timing.get("very_low_confidence_ratio")) or 0.0),
        -float(timing.get("effective_low_confidence_ratio", timing.get("low_confidence_ratio")) or 0.0),
        -float(timing.get("max_tail_gap") or 0.0),
        -float(timing.get("max_unexplained_internal_gap") or timing.get("max_internal_gap") or 0.0),
        -float(timing.get("max_internal_gap") or 0.0),
        -int(attempt.trim_count),
        float(attempt.result.avg_token_confidence),
        float(attempt.result.normalized_log_prob),
    )


def _attempt_note(attempt: _AlignmentAttempt) -> str:
    timing = attempt.timing_summary
    confidence_summary = timing.get("confidence_issue_summary") or {}
    confidence_scope = confidence_summary.get("effective_scope") or confidence_summary.get("scope")
    raw_confidence_scope = confidence_summary.get("scope")
    effective_very_low_ratio = float(
        timing.get("effective_very_low_confidence_ratio", timing.get("very_low_confidence_ratio")) or 0.0
    )
    pause_compression = (attempt.result.alignment_debug or {}).get("pause_compression") or {}
    return (
        f"{attempt.name}: gate_passed={attempt.confidence.passed}; "
        f"quality={timing.get('quality_level')}; "
        f"coverage={attempt.result.coverage:.3f}; "
        f"avg_conf={attempt.result.avg_token_confidence:.3f}; "
        f"very_low_ratio={float(timing.get('very_low_confidence_ratio') or 0.0):.3f}; "
        f"effective_very_low_ratio={effective_very_low_ratio:.3f}; "
        f"confidence_scope={confidence_scope}; "
        f"raw_confidence_scope={raw_confidence_scope}; "
        f"max_internal_gap={float(timing.get('max_internal_gap') or 0.0):.3f}; "
        f"max_unexplained_internal_gap={float(timing.get('max_unexplained_internal_gap') or 0.0):.3f}; "
        f"pause_gap_count={int(timing.get('pause_gap_count') or 0)}; "
        f"max_tail_gap={float(timing.get('max_tail_gap') or 0.0):.3f}; "
        f"pause_frames_removed={int(pause_compression.get('removed_frame_count') or 0)}"
    )


def _large_tail_trim_has_clear_gain(attempt: _AlignmentAttempt, full_attempt: _AlignmentAttempt) -> bool:
    if attempt.trim_count <= 2:
        return True

    full_very_low_ratio = float(
        full_attempt.timing_summary.get(
            "effective_very_low_confidence_ratio",
            full_attempt.timing_summary.get("very_low_confidence_ratio"),
        )
        or 0.0
    )
    attempt_very_low_ratio = float(
        attempt.timing_summary.get(
            "effective_very_low_confidence_ratio",
            attempt.timing_summary.get("very_low_confidence_ratio"),
        )
        or 0.0
    )
    very_low_improved = full_very_low_ratio - attempt_very_low_ratio >= 0.05
    tail_gap_removed = (
        _tail_region_max_gap(full_attempt.result, attempt.trim_count) > 0.30
        and float(attempt.timing_summary.get("max_tail_gap") or 0.0) <= 0.30
    )
    return very_low_improved or tail_gap_removed


def _tail_attempt_is_selectable(attempt: _AlignmentAttempt, full_attempt: _AlignmentAttempt) -> bool:
    if attempt.trim_count <= 0:
        return False
    if not attempt.confidence.passed:
        return False
    if not _large_tail_trim_allowed(full_attempt, attempt.trim_count):
        return False
    if not _large_tail_trim_has_clear_gain(attempt, full_attempt):
        return False
    return True


def _select_alignment_attempt(attempts: list[_AlignmentAttempt]) -> _AlignmentAttempt:
    full_attempt = attempts[0]
    selectable_tail_attempts = [
        attempt
        for attempt in attempts[1:]
        if _tail_attempt_is_selectable(attempt, full_attempt)
    ]
    if not selectable_tail_attempts:
        return full_attempt
    return max(selectable_tail_attempts, key=_alignment_attempt_rank)


def forced_alignment_stage(ctx: PipelineContext) -> EvaluationResult | None:
    """Align the selected reference IPA to frame-level CTC logits."""

    assert ctx.coarse_alignment is not None and ctx.recognition is not None
    label_to_id, blank_id = _get_alignment_vocab(ctx.recognizer)
    ctx.alignment_blank_id = blank_id
    evidence = _alignment_evidence(ctx)
    candidate = ctx.coarse_alignment.selected_reference_candidate
    attempts: list[_AlignmentAttempt] = []

    try:
        full_alignment = force_align_candidate(
            candidate,
            ctx.recognition.logits,
            ctx.recognition.frame_timestamps or [],
            label_to_id,
            blank_id,
            frame_confidence=ctx.recognition.frame_confidence,
            frame_energy=ctx.recognition.frame_energy,
        )
        full_attempt = _make_alignment_attempt("full", full_alignment, evidence=evidence)
        attempts.append(full_attempt)
        ctx.alignment_notes.append(f"Forced alignment attempt result: {_attempt_note(full_attempt)}")

        if not full_attempt.confidence.passed:
            heuristic_trim_count = _unstable_tail_trim_count(full_attempt.result)
            trim_counts = _tail_retry_trim_counts(candidate, full_attempt)
            if trim_counts:
                ctx.alignment_notes.append(
                    "Forced alignment iterative tail retry requested: "
                    f"trim_counts={trim_counts}; "
                    f"heuristic_trim_count={heuristic_trim_count}; "
                    f"initial_confidence={full_attempt.confidence.message}"
                )
            for trim_count in trim_counts:
                retry_candidate = _trim_candidate_tail(candidate, trim_count)
                try:
                    retry_alignment = force_align_candidate(
                        retry_candidate,
                        ctx.recognition.logits,
                        ctx.recognition.frame_timestamps or [],
                        label_to_id,
                        blank_id,
                        frame_confidence=ctx.recognition.frame_confidence,
                        frame_energy=ctx.recognition.frame_energy,
                    )
                    retry_attempt = _make_alignment_attempt(
                        f"tail_trim_{trim_count}",
                        retry_alignment,
                        trim_count=trim_count,
                        evidence=evidence,
                    )
                    attempts.append(retry_attempt)
                    ctx.alignment_notes.append(
                        "Forced alignment iterative tail retry result: "
                        f"retry_confidence={retry_attempt.confidence.message}; "
                        f"{_attempt_note(retry_attempt)}"
                    )
                except ValueError as retry_exc:
                    ctx.alignment_notes.append(
                        f"Forced alignment tail retry failed: trim_count={trim_count}; error={retry_exc}"
                    )
            if not trim_counts:
                ctx.alignment_notes.append(
                    "Forced alignment tail retry skipped: "
                    "failure was not isolated to unstable final phone(s); "
                    f"max_internal_gap={full_attempt.timing_summary['max_internal_gap']:.3f}; "
                    f"max_unexplained_internal_gap={full_attempt.timing_summary['max_unexplained_internal_gap']:.3f}; "
                    f"max_tail_gap={full_attempt.timing_summary['max_tail_gap']:.3f}"
                )

        selected_attempt = _select_alignment_attempt(attempts)
        ctx.forced_alignment = selected_attempt.result
        ctx.alignment_notes.append(
            "Forced alignment selected attempt: "
            f"{selected_attempt.name}; {_attempt_note(selected_attempt)}"
        )
        if selected_attempt.trim_count:
            excluded = candidate.ipa.tokens[-selected_attempt.trim_count:]
            excluded_text = " ".join(token.symbol for token in excluded)
            ctx.alignment_notes.append(
                f"Retried forced alignment after excluding unstable final phone(s): {excluded_text}"
            )
        elif not full_attempt.confidence.passed and len(attempts) > 1:
            ctx.alignment_notes.append(
                "Forced alignment tail retry candidates were rejected; "
                "keeping full alignment because no tail-trim candidate passed the gate "
                "with sufficient improvement."
            )
    except ValueError as exc:
        ctx.acceptance_notes.append(
            "Forced alignment failed but evaluation continued under soft-accept policy. "
            f"error={exc}"
        )
        ctx.alignment_notes.append(f"Forced alignment failed: {exc}")
    return None


def alignment_confidence_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Record forced-alignment confidence failures while keeping accepted output."""

    assert ctx.coarse_alignment is not None and ctx.forced_alignment is not None and ctx.recognition is not None

    ctx.alignment_confidence = assess_alignment_confidence(
        ctx.forced_alignment,
        **_alignment_evidence(ctx),
    )

    if ctx.alignment_confidence.passed:
        return None

    ctx.acceptance_notes.append(
        "Forced alignment confidence gate failed but evaluation continued under soft-accept policy. "
        f"message={ctx.alignment_confidence.message}"
    )
    return None

def scoring_and_error_stage(ctx: PipelineContext) -> None:
    """Create score breakdown and pronunciation issue analysis."""

    assert ctx.coarse_alignment is not None
    ctx.score_breakdown = build_score_breakdown(ctx.coarse_alignment)
    issues = classify_alignment_errors(ctx.coarse_alignment, profile=ctx.profile)
    ctx.feedback_report = build_feedback_report(issues, ctx.score_breakdown, profile=ctx.profile)


def make_ready_result(ctx: PipelineContext) -> EvaluationResult:
    """Create the final successful evaluation result."""

    assert (
        ctx.coarse_alignment is not None
        and ctx.recognition is not None
    )
    audio_gate_passed = ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None
    alignment_gate_passed = ctx.alignment_confidence.passed if ctx.alignment_confidence is not None else False
    raw_coarse_gate_passed = ctx.coarse_alignment.normalized_score >= ctx.coarse_similarity_threshold
    coarse_gate_recovered_by_alignment = (not raw_coarse_gate_passed) and alignment_gate_passed
    coarse_gate_passed = raw_coarse_gate_passed or coarse_gate_recovered_by_alignment
    all_gates_passed = (
        audio_gate_passed is not False
        and coarse_gate_passed
        and alignment_gate_passed
    )
    status_message = (
        "Audio quality, coarse similarity, and forced alignment confidence gates passed."
        if all_gates_passed
        else "Accepted with warnings; one or more quality/alignment gates failed under soft-accept policy."
    )
    return make_evaluation_result(
        ctx,
        status="ready",
        message=status_message,
        alignment_result=ctx.coarse_alignment,
        score_breakdown=ctx.score_breakdown,
        feedback_report=ctx.feedback_report,
        forced_alignment_result=ctx.forced_alignment,
        alignment_confidence_report=ctx.alignment_confidence,
        debug={
            "audio_quality_gate_passed": audio_gate_passed,
            "coarse_token_alignment_gate_passed": coarse_gate_passed,
            "raw_coarse_token_alignment_gate_passed": raw_coarse_gate_passed,
            "coarse_gate_recovered_by_alignment": coarse_gate_recovered_by_alignment,
            "alignment_confidence_gate_passed": alignment_gate_passed,
            "soft_accept_policy_applied": not all_gates_passed,
            "coarse_similarity": ctx.coarse_alignment.normalized_score,
            "coarse_similarity_threshold": ctx.coarse_similarity_threshold,
            "frame_confidence_preview": ctx.recognition.frame_confidence[:10] if ctx.recognition.frame_confidence else [],
            "frame_timestamp_preview": ctx.recognition.frame_timestamps[:10] if ctx.recognition.frame_timestamps else [],
            "selected_candidate_notes": ctx.coarse_alignment.selected_reference_candidate.notes,
            "score_source": "coarse_token_alignment",
            "score_is_final": True,
            "score_available_for_debug": True,
        },
    )


def make_evaluation_result(
    ctx: PipelineContext,
    *,
    status: str,
    message: str,
    alignment_result: AlignmentResult | None = None,
    score_breakdown: ScoreBreakdown | None = None,
    feedback_report: FeedbackReport | None = None,
    forced_alignment_result: ForcedAlignmentResult | None = None,
    alignment_confidence_report: AlignmentConfidenceReport | None = None,
    debug: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Create the shared EvaluationResult dataclass used by exporters."""

    assert ctx.reference is not None and ctx.recognition is not None and ctx.selected_candidate is not None
    debug_payload = {
        "raw_label_text": ctx.recognition.raw_label_text,
        "raw_labels": ctx.recognition.raw_labels,
        "audio_trim": {
            "start_sec": ctx.recognition.trim_start_sec,
            "end_sec": ctx.recognition.trim_end_sec,
            "original_duration_sec": ctx.recognition.original_duration_sec,
            "trimmed_duration_sec": ctx.recognition.trimmed_duration_sec,
        },
    }
    if debug:
        debug_payload.update(debug)
    if ctx.alignment_notes:
        debug_payload["alignment_notes"] = ctx.alignment_notes
    if ctx.acceptance_notes:
        debug_payload["acceptance_notes"] = ctx.acceptance_notes
    if ctx.stage_latencies_ms:
        debug_payload["stage_latencies_ms"] = dict(ctx.stage_latencies_ms)
    alignment_for_debug = forced_alignment_result or ctx.forced_alignment
    if alignment_for_debug is not None:
        timing_summary = summarize_alignment_timing(
            alignment_for_debug,
            **_alignment_evidence(ctx),
        )
        alignment_gate_passed = debug_payload.get("alignment_confidence_gate_passed")
        if alignment_gate_passed is None and alignment_confidence_report is not None:
            alignment_gate_passed = alignment_confidence_report.passed
        debug_payload["alignment_timing_warnings"] = timing_summary
        prosody_usage = decide_prosody_alignment_usage(
            timing_summary,
            evaluation_status=status,
            alignment_gate_passed=alignment_gate_passed,
        )
        soft_accept_limited_by: list[str] = []
        if debug_payload.get("audio_quality_gate_passed") is False:
            soft_accept_limited_by.append("audio_quality")
        if debug_payload.get("coarse_token_alignment_gate_passed") is False:
            soft_accept_limited_by.append("coarse_alignment")
        if soft_accept_limited_by:
            prosody_usage = dict(prosody_usage)
            reasons = list(prosody_usage.get("reasons") or [])
            for reason in soft_accept_limited_by:
                reason_key = f"{reason}_gate_failed"
                if reason_key not in reasons:
                    reasons.append(reason_key)
            if "soft_accept_policy_applied" not in reasons:
                reasons.append("soft_accept_policy_applied")
            prosody_usage["reasons"] = reasons
            limited_by = list(prosody_usage.get("limited_by") or [])
            for name in soft_accept_limited_by:
                if name not in limited_by:
                    limited_by.append(name)
            prosody_usage["limited_by"] = limited_by
            limitation_causes = dict(prosody_usage.get("limitation_causes") or {})
            for name in soft_accept_limited_by:
                limitation_causes[name] = True
            prosody_usage["limitation_causes"] = limitation_causes
            if prosody_usage.get("recommended_usage") == "full":
                prosody_usage["recommended_usage"] = "cautious"
        debug_payload["prosody_alignment_usage"] = prosody_usage
    elif status == "ready":
        debug_payload["prosody_alignment_usage"] = decide_prosody_alignment_usage(
            None,
            evaluation_status=status,
            alignment_gate_passed=False,
        )

    return EvaluationResult(
        evaluation_status=status,
        status_message=message,
        reference_text=ctx.reference.normalized_text,
        reference_pronunciation=ctx.reference.representative_pronunciation,
        reference_ipa=ctx.reference.representative_ipa,
        reference_candidates=ctx.reference.candidates,
        selected_reference_candidate=ctx.selected_candidate,
        user_ipa_raw=ctx.recognition.raw_text,
        user_ipa_normalized=ctx.recognition.normalized_text,
        user_tokens=ctx.recognition.tokens,
        alignment_result=alignment_result,
        score_breakdown=score_breakdown,
        feedback_report=feedback_report,
        quality_report=ctx.recognition.quality_report,
        forced_alignment_result=forced_alignment_result,
        alignment_confidence_report=alignment_confidence_report,
        debug=debug_payload,
    )


def _time_stage(ctx: PipelineContext, key: str, func):
    start = time.perf_counter()
    try:
        return func()
    finally:
        ctx.stage_latencies_ms[key] = (time.perf_counter() - start) * 1000.0


def _attach_stage_latencies(result: EvaluationResult, ctx: PipelineContext) -> EvaluationResult:
    result.debug["stage_latencies_ms"] = dict(ctx.stage_latencies_ms)
    return result


def run_evaluation(
    audio_path: str | Path,
    reference_text: str,
    *,
    recognizer: AudioToIPARecognizer | None = None,
    profile: str = "ru",
    coarse_similarity_threshold: float = COARSE_SIMILARITY_THRESHOLD,
) -> EvaluationResult:
    """Run the full evaluation flow and return the internal result object."""

    ctx = PipelineContext(
        audio_path=Path(audio_path),
        reference_text=reference_text,
        profile=profile,
        coarse_similarity_threshold=coarse_similarity_threshold,
        recognizer=recognizer or get_default_recognizer(),
        alignment_calibration_stats=load_confidence_calibration(),
    )

    _time_stage(ctx, "build_reference_ms", lambda: build_reference_stage(ctx))
    _time_stage(ctx, "recognize_audio_ms", lambda: recognize_audio_stage(ctx))

    early_result = audio_quality_gate(ctx) or recognition_gate(ctx)
    if early_result is not None:
        return _attach_stage_latencies(early_result, ctx)

    _time_stage(ctx, "coarse_alignment_ms", lambda: coarse_token_alignment_stage(ctx))
    early_result = coarse_token_alignment_gate(ctx)
    if early_result is not None:
        return _attach_stage_latencies(early_result, ctx)

    early_result = _time_stage(ctx, "forced_alignment_ms", lambda: forced_alignment_stage(ctx))
    if early_result is not None:
        return _attach_stage_latencies(early_result, ctx)

    if ctx.forced_alignment is not None:
        early_result = _time_stage(ctx, "confidence_gate_ms", lambda: alignment_confidence_gate(ctx))
        if early_result is not None:
            return _attach_stage_latencies(early_result, ctx)
    else:
        ctx.stage_latencies_ms["confidence_gate_ms"] = 0.0

    _time_stage(ctx, "scoring_ms", lambda: scoring_and_error_stage(ctx))
    return make_ready_result(ctx)


def build_backend_payload(
    result: EvaluationResult,
    *,
    audio_path: str | Path,
    profile: str = "ru",
    artifact_paths: dict[str, Path | None] | None = None,
) -> dict[str, Any]:
    """Build route-friendly outputs from the internal result object."""

    audio_path = Path(audio_path)
    artifact_dir = artifact_paths.get("artifact_dir") if artifact_paths else None
    json_path = artifact_paths.get("json_path") if artifact_paths else None
    saved_audio_path = artifact_paths.get("audio_path") if artifact_paths else None
    payload = build_evaluation_backdata(
        result,
        profile=profile,
        audio_source_name=audio_path.name,
        artifact_id=artifact_dir.name if artifact_dir is not None else None,
        json_file_name=json_path.name if json_path is not None else None,
        audio_file_name=saved_audio_path.name if saved_audio_path is not None else None,
    )

    payload["prosody_input"]["audio_file_path"] = (
        str(saved_audio_path.resolve()) if saved_audio_path is not None else str(audio_path.resolve())
    )
    payload["prosody_input"]["audio_trim"] = payload.get("debug", {}).get("audio_trim")
    payload["prosody_input"]["timing_reference"] = (
        "trimmed_artifact_audio" if saved_audio_path is not None else "trimmed_audio_coordinates"
    )
    payload["prosody_input"]["reference_phonemes"] = [
        {
            "index": index,
            "token": token.symbol,
            "category": token.category,
            "syllable_position": token.syllable_position,
        }
        for index, token in enumerate(result.selected_reference_candidate.ipa.tokens)
    ]
    if result.forced_alignment_result is not None:
        aligned_count = len(result.forced_alignment_result.token_symbols)
        reference_tokens = result.selected_reference_candidate.ipa.tokens
        if aligned_count < len(reference_tokens):
            payload["prosody_input"]["excluded_reference_tail"] = [
                {
                    "index": index,
                    "token": token.symbol,
                    "category": token.category,
                    "syllable_position": token.syllable_position,
                    "reason": "unstable_forced_alignment_tail",
                }
                for index, token in enumerate(reference_tokens[aligned_count:], start=aligned_count)
            ]
            payload["prosody_input"]["alignment_is_partial"] = True
        else:
            payload["prosody_input"]["excluded_reference_tail"] = []
            payload["prosody_input"]["alignment_is_partial"] = False

        usage = payload["prosody_input"].get("alignment_usage")
        if usage is not None and payload["prosody_input"]["alignment_is_partial"]:
            reasons = list(usage.get("reasons") or [])
            if "partial_alignment_tail_excluded" not in reasons:
                reasons.append("partial_alignment_tail_excluded")
            usage["reasons"] = reasons
            limited_by = list(usage.get("limited_by") or [])
            if "partial_tail" not in limited_by:
                limited_by.append("partial_tail")
            usage["limited_by"] = limited_by
            limitation_causes = dict(usage.get("limitation_causes") or {})
            limitation_causes["partial_tail"] = True
            usage["limitation_causes"] = limitation_causes
            if usage.get("recommended_usage") == "full":
                usage["recommended_usage"] = "cautious"
            usage["is_partial_alignment"] = True
        elif usage is not None:
            usage["is_partial_alignment"] = False
            limitation_causes = dict(usage.get("limitation_causes") or {})
            limitation_causes["partial_tail"] = False
            usage["limitation_causes"] = limitation_causes
        if usage is not None:
            payload["prosody_input"]["alignment_for_prosody"] = usage["alignment_for_prosody"]
            payload["prosody_input"]["detailed_timing_allowed"] = usage["detailed_timing_allowed"]
            payload["prosody_input"]["prosody_recommended_usage"] = usage["recommended_usage"]
    else:
        payload["prosody_input"]["excluded_reference_tail"] = []
        payload["prosody_input"]["alignment_is_partial"] = False

    return {
        "status": payload["status"],
        "gates": payload["gates"],
        "llm_feedback_input": payload["llm_feedback_input"],
        "prosody_input": payload["prosody_input"],
        "full_payload": payload,
        "artifact_paths": {
            "artifact_dir": str(artifact_dir.resolve()) if artifact_dir is not None else None,
            "json_path": str(json_path.resolve()) if json_path is not None else None,
            "audio_path": str(saved_audio_path.resolve()) if saved_audio_path is not None else None,
        },
    }


def save_backend_artifacts(
    result: EvaluationResult,
    *,
    audio_path: str | Path,
    profile: str,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Path | None]:
    """Save JSON and source audio under artifacts/<timestamp>/."""

    audio_path = Path(audio_path)
    return save_evaluation_bundle(
        result,
        profile=profile,
        audio_source_name=audio_path.name,
        source_audio_path=audio_path,
        out_dir=Path(artifacts_dir) if artifacts_dir is not None else None,
    )


def evaluate_pronunciation_file(
    audio_path: str | Path,
    reference_text: str,
    *,
    profile: str = "ru",
    recognizer: AudioToIPARecognizer | None = None,
    save_artifacts: bool = True,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """One-call API helper for backend integration."""

    audio_path = Path(audio_path)
    result = run_evaluation(
        audio_path,
        reference_text,
        recognizer=recognizer,
        profile=profile,
    )

    artifact_paths = None
    if save_artifacts:
        artifact_paths = save_backend_artifacts(
            result,
            audio_path=audio_path,
            profile=profile,
            artifacts_dir=artifacts_dir,
        )

    backend_payload = build_backend_payload(
        result,
        audio_path=audio_path,
        profile=profile,
        artifact_paths=artifact_paths,
    )
    if artifact_paths and artifact_paths.get("json_path") is not None:
        artifact_paths["json_path"].write_text(
            json.dumps(backend_payload["full_payload"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return backend_payload


def get_llm_feedback_input(audio_path: str | Path, reference_text: str, **kwargs) -> dict[str, Any]:
    """Convenience wrapper for a feedback-LLM API route."""

    return evaluate_pronunciation_file(audio_path, reference_text, **kwargs)["llm_feedback_input"]


def get_prosody_input(audio_path: str | Path, reference_text: str, **kwargs) -> dict[str, Any]:
    """Convenience wrapper for a prosody-analysis API route."""

    return evaluate_pronunciation_file(audio_path, reference_text, **kwargs)["prosody_input"]
