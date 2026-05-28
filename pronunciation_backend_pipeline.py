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
from typing import Any


from src import cost_model
from src.alignment import score_reference_candidates
from src.audio_to_ipa import AudioToIPARecognizer
from src.backdata_export import build_evaluation_backdata, save_evaluation_bundle
from src.error_analysis import classify_alignment_errors
from src.feedback_report import build_feedback_report
from src.forced_alignment import force_align_candidate
from src.quality import assess_alignment_confidence
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
    """Stop early when the audio is too short, silent, or clipped."""

    assert ctx.reference is not None and ctx.recognition is not None and ctx.selected_candidate is not None
    report = ctx.recognition.quality_report
    if report is None or report.passed:
        return None

    return make_evaluation_result(
        ctx,
        status="retry",
        message="Audio quality gate failed. Re-recording is recommended.",
        debug={
            "audio_quality_gate_passed": False,
            "coarse_token_alignment_gate_passed": None,
            "alignment_confidence_gate_passed": None,
        },
    )


def recognition_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Stop early when no stable phone/IPA sequence was decoded."""

    assert ctx.reference is not None and ctx.recognition is not None
    if ctx.recognition.tokens:
        return None

    return make_evaluation_result(
        ctx,
        status="retry",
        message="Could not extract a stable phone/IPA sequence from the audio.",
        debug={
            "audio_quality_gate_passed": ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None,
            "coarse_token_alignment_gate_passed": None,
            "alignment_confidence_gate_passed": None,
        },
    )


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
    """Filter out utterances that are too different from the reference sentence."""

    assert ctx.coarse_alignment is not None and ctx.recognition is not None
    if ctx.coarse_alignment.normalized_score >= ctx.coarse_similarity_threshold:
        return None

    return make_evaluation_result(
        ctx,
        status="retry",
        message="Coarse IPA similarity gate failed. The utterance may not match the reference sentence.",
        alignment_result=ctx.coarse_alignment,
        debug={
            "audio_quality_gate_passed": ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None,
            "coarse_token_alignment_gate_passed": False,
            "alignment_confidence_gate_passed": None,
            "coarse_similarity": ctx.coarse_alignment.normalized_score,
            "coarse_similarity_threshold": ctx.coarse_similarity_threshold,
        },
    )


def _unstable_tail_trim_count(result: ForcedAlignmentResult) -> int:
    segments = result.segments
    if len(segments) < 3:
        return 0

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

    if segments[-1].confidence < 0.005:
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


def forced_alignment_stage(ctx: PipelineContext) -> EvaluationResult | None:
    """Align the selected reference IPA to frame-level CTC logits."""

    assert ctx.coarse_alignment is not None and ctx.recognition is not None
    tokenizer = getattr(ctx.recognizer.processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Recognizer processor does not expose a tokenizer for forced alignment.")

    try:
        ctx.forced_alignment = force_align_candidate(
            ctx.coarse_alignment.selected_reference_candidate,
            ctx.recognition.logits or [],
            ctx.recognition.frame_timestamps or [],
            tokenizer.get_vocab(),
            tokenizer.pad_token_id,
        )
        initial_confidence = assess_alignment_confidence(ctx.forced_alignment)
        trim_count = _unstable_tail_trim_count(ctx.forced_alignment)
        if not initial_confidence.passed and trim_count:
            retry_candidate = _trim_candidate_tail(
                ctx.coarse_alignment.selected_reference_candidate,
                trim_count,
            )
            retry_alignment = force_align_candidate(
                retry_candidate,
                ctx.recognition.logits or [],
                ctx.recognition.frame_timestamps or [],
                tokenizer.get_vocab(),
                tokenizer.pad_token_id,
            )
            retry_confidence = assess_alignment_confidence(retry_alignment)
            if retry_confidence.passed:
                excluded = ctx.coarse_alignment.selected_reference_candidate.ipa.tokens[-trim_count:]
                excluded_text = " ".join(token.symbol for token in excluded)
                ctx.forced_alignment = retry_alignment
                ctx.alignment_notes.append(
                    f"Retried forced alignment after excluding unstable final phone(s): {excluded_text}"
                )
    except ValueError as exc:
        return make_evaluation_result(
            ctx,
            status="discarded",
            message="Forced alignment failed because the reference contains an unsupported alignment symbol.",
            alignment_result=ctx.coarse_alignment,
            debug={
                "audio_quality_gate_passed": ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None,
                "coarse_token_alignment_gate_passed": True,
                "alignment_confidence_gate_passed": None,
                "coarse_similarity": ctx.coarse_alignment.normalized_score,
                "coarse_similarity_threshold": ctx.coarse_similarity_threshold,
                "alignment_error": str(exc),
            },
        )
    return None


def alignment_confidence_gate(ctx: PipelineContext) -> EvaluationResult | None:
    """Check whether the frame-level forced alignment is reliable enough."""

    assert ctx.coarse_alignment is not None and ctx.forced_alignment is not None and ctx.recognition is not None

    ctx.alignment_confidence = assess_alignment_confidence(ctx.forced_alignment)

    if ctx.alignment_confidence.passed:
        return None

    # Forced alignment가 실패해도 coarse alignment 기반 점수는 분석용으로 저장한다.
    # 단, 이 점수는 최종 점수가 아니라 debug/reference용이다.
    ctx.score_breakdown = build_score_breakdown(ctx.coarse_alignment)
    issues = classify_alignment_errors(ctx.coarse_alignment, profile=ctx.profile)
    ctx.feedback_report = build_feedback_report(issues, ctx.score_breakdown, profile=ctx.profile)

    return make_evaluation_result(
        ctx,
        status="discarded",
        message=(
            "Forced alignment confidence gate failed. "
            "Coarse score is provided for debugging, but should not be used as final pronunciation score."
        ),
        alignment_result=ctx.coarse_alignment,
        forced_alignment_result=ctx.forced_alignment,
        alignment_confidence_report=ctx.alignment_confidence,
        score_breakdown=ctx.score_breakdown,
        feedback_report=ctx.feedback_report,
        debug={
            "audio_quality_gate_passed": ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None,
            "coarse_token_alignment_gate_passed": True,
            "alignment_confidence_gate_passed": False,
            "coarse_similarity": ctx.coarse_alignment.normalized_score,
            "coarse_similarity_threshold": ctx.coarse_similarity_threshold,
            "score_source": "coarse_token_alignment",
            "score_is_final": False,
            "score_available_for_debug": True,
        },
    )


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
        and ctx.forced_alignment is not None
        and ctx.alignment_confidence is not None
        and ctx.recognition is not None
    )
    return make_evaluation_result(
        ctx,
        status="ready",
        message="Audio quality, coarse similarity, and forced alignment confidence gates passed.",
        alignment_result=ctx.coarse_alignment,
        score_breakdown=ctx.score_breakdown,
        feedback_report=ctx.feedback_report,
        forced_alignment_result=ctx.forced_alignment,
        alignment_confidence_report=ctx.alignment_confidence,
        debug={
            "audio_quality_gate_passed": ctx.recognition.quality_report.passed if ctx.recognition.quality_report else None,
            "coarse_token_alignment_gate_passed": True,
            "alignment_confidence_gate_passed": True,
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
    }
    if debug:
        debug_payload.update(debug)
    if ctx.alignment_notes:
        debug_payload["alignment_notes"] = ctx.alignment_notes

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
    )

    build_reference_stage(ctx)
    recognize_audio_stage(ctx)

    early_result = audio_quality_gate(ctx) or recognition_gate(ctx)
    if early_result is not None:
        return early_result

    coarse_token_alignment_stage(ctx)
    early_result = coarse_token_alignment_gate(ctx)
    if early_result is not None:
        return early_result

    early_result = forced_alignment_stage(ctx)
    if early_result is not None:
        return early_result

    early_result = alignment_confidence_gate(ctx)
    if early_result is not None:
        return early_result

    scoring_and_error_stage(ctx)
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
