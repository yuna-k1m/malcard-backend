from __future__ import annotations

"""Serialize evaluation outputs for downstream LLM feedback and prosody analysis."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.types import (
    AlignmentConfidenceReport,
    AlignmentResult,
    AlignmentSegment,
    AudioQualityReport,
    EvaluationResult,
    FeedbackReport,
    ForcedAlignmentResult,
    IPASequence,
    PhoneToken,
    PronunciationCandidate,
    PronunciationIssue,
    ScoreBreakdown,
)


SCHEMA_VERSION = "1.2"
DEFAULT_OUTPUT_DIR = Path("artifacts")


def _serialize_phone_token(token: PhoneToken) -> dict:
    return {
        "symbol": token.symbol,
        "category": token.category,
        "syllable_position": token.syllable_position,
        "confidence": token.confidence,
    }


def _serialize_ipa_sequence(sequence: IPASequence) -> dict:
    return {
        "raw_text": sequence.raw_text,
        "normalized_text": sequence.normalized_text,
        "tokens": [_serialize_phone_token(token) for token in sequence.tokens],
        "token_symbols": sequence.token_symbols,
    }


def _serialize_candidate(candidate: PronunciationCandidate) -> dict:
    return {
        "pronunciation": candidate.pronunciation,
        "ipa": _serialize_ipa_sequence(candidate.ipa),
        "notes": candidate.notes,
        "is_primary": candidate.is_primary,
    }


def _serialize_audio_quality(report: AudioQualityReport | None) -> dict | None:
    if report is None:
        return None
    return {
        "passed": report.passed,
        "duration_sec": report.duration_sec,
        "rms_db": report.rms_db,
        "silence_ratio": report.silence_ratio,
        "clipping_ratio": report.clipping_ratio,
        "reasons": report.reasons,
    }


def _serialize_alignment_step(step) -> dict:
    return {
        "op": step.op,
        "ref_token": _serialize_phone_token(step.ref_token) if step.ref_token is not None else None,
        "hyp_token": _serialize_phone_token(step.hyp_token) if step.hyp_token is not None else None,
        "cost": step.cost,
        "error_type": step.error_type,
        "detail": step.detail,
        "ref_index": step.ref_index,
        "hyp_index": step.hyp_index,
        "feature_penalties": step.feature_penalties,
    }


def _serialize_alignment_result(result: AlignmentResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "total_cost": result.total_cost,
        "max_cost": result.max_cost,
        "normalized_score": result.normalized_score,
        "aligned_ref": result.aligned_ref,
        "aligned_hyp": result.aligned_hyp,
        "segment_errors": result.segment_errors,
        "feature_penalties": result.feature_penalties,
        "selected_reference_candidate": _serialize_candidate(result.selected_reference_candidate),
        "steps": [_serialize_alignment_step(step) for step in result.ops],
    }


def _serialize_score_breakdown(score: ScoreBreakdown | None) -> dict | None:
    if score is None:
        return None
    return {
        "overall": score.overall,
        "consonant": score.consonant,
        "vowel": score.vowel,
        "coda": score.coda,
        "fluency_like": score.fluency_like,
        "raw_cost": score.raw_cost,
        "max_cost": score.max_cost,
        "penalty_summary": score.penalty_summary,
    }


def _serialize_issue(issue: PronunciationIssue) -> dict:
    return {
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "description": issue.description,
        "tip": issue.tip,
        "ref_token": issue.ref_token,
        "hyp_token": issue.hyp_token,
        "cost": issue.cost,
        "acceptable": issue.acceptable,
        "debug": issue.debug,
    }


def _serialize_feedback_report(report: FeedbackReport | None) -> dict | None:
    if report is None:
        return None
    return {
        "summary_level": report.summary_level,
        "headline": report.headline,
        "issues": [_serialize_issue(issue) for issue in report.issues],
        "tips": report.tips,
        "debug_notes": report.debug_notes,
    }


def _serialize_alignment_segment(segment: AlignmentSegment) -> dict:
    return {
        "token": segment.token,
        "label": segment.label,
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "duration": max(0.0, segment.end_time - segment.start_time),
        "frame_start": segment.frame_start,
        "frame_end": segment.frame_end,
        "confidence": segment.confidence,
    }


def _serialize_forced_alignment(result: ForcedAlignmentResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "labels": result.labels,
        "token_symbols": result.token_symbols,
        "segments": [_serialize_alignment_segment(segment) for segment in result.segments],
        "total_log_prob": result.total_log_prob,
        "normalized_log_prob": result.normalized_log_prob,
        "avg_token_confidence": result.avg_token_confidence,
        "coverage": result.coverage,
        "blank_ratio": result.blank_ratio,
    }


def _serialize_alignment_confidence(report: AlignmentConfidenceReport | None) -> dict | None:
    if report is None:
        return None
    return {
        "passed": report.passed,
        "avg_token_confidence": report.avg_token_confidence,
        "coverage": report.coverage,
        "normalized_log_prob": report.normalized_log_prob,
        "message": report.message,
    }


def _infer_coarse_gate_passed(result: EvaluationResult) -> bool | None:
    if result.alignment_result is None:
        return None
    if result.forced_alignment_result is not None:
        return True
    if result.alignment_confidence_report is not None:
        return True
    if result.score_breakdown is not None or result.feedback_report is not None:
        return True
    return False


def _build_gate_summary(result: EvaluationResult) -> dict:
    quality_payload = _serialize_audio_quality(result.quality_report)
    coarse_passed = result.debug.get("coarse_token_alignment_gate_passed")
    if coarse_passed is None:
        coarse_passed = _infer_coarse_gate_passed(result)
    alignment_payload = _serialize_alignment_confidence(result.alignment_confidence_report)
    return {
        "audio_quality_gate": {
            "performed": result.quality_report is not None,
            "passed": result.debug.get("audio_quality_gate_passed", result.quality_report.passed if result.quality_report is not None else None),
            "report": quality_payload,
        },
        "coarse_token_alignment_gate": {
            "performed": result.alignment_result is not None,
            "passed": coarse_passed,
            "normalized_score": result.alignment_result.normalized_score if result.alignment_result is not None else None,
        },
        "alignment_confidence_gate": {
            "performed": result.alignment_confidence_report is not None,
            "passed": result.debug.get(
                "alignment_confidence_gate_passed",
                result.alignment_confidence_report.passed if result.alignment_confidence_report is not None else None,
            ),
            "report": alignment_payload,
        },
    }


def _resolve_audio_suffix(audio_source_name: str | None, source_audio_path: str | Path | None) -> str:
    for candidate in (audio_source_name, source_audio_path):
        if not candidate:
            continue
        suffix = Path(candidate).suffix
        if suffix:
            return suffix
    return ".wav"


def build_evaluation_backdata(
    result: EvaluationResult,
    *,
    profile: str,
    audio_source_name: str | None = None,
    artifact_id: str | None = None,
    json_file_name: str | None = None,
    audio_file_name: str | None = None,
) -> dict:
    timestamp = datetime.now().astimezone().isoformat()
    feedback_payload = _serialize_feedback_report(result.feedback_report)
    mismatch_steps = []
    if result.alignment_result is not None:
        for step in result.alignment_result.ops:
            if step.cost <= 0:
                continue
            mismatch_steps.append(
                {
                    "ref": step.ref_token.symbol if step.ref_token is not None else "∅",
                    "hyp": step.hyp_token.symbol if step.hyp_token is not None else "∅",
                    "cost": step.cost,
                    "error_type": step.error_type,
                    "detail": step.detail,
                }
            )

    gate_summary = _build_gate_summary(result)
    llm_feedback_input = {
        "reference_text": result.reference_text,
        "reference_pronunciation": result.reference_pronunciation,
        "reference_ipa": result.reference_ipa.normalized_text,
        "user_ipa": result.user_ipa_normalized,
        "status": result.evaluation_status,
        "status_message": result.status_message,
        "score_breakdown": _serialize_score_breakdown(result.score_breakdown),
        "gate_summary": gate_summary,
        "mismatches": mismatch_steps,
        "issues": feedback_payload["issues"] if feedback_payload is not None else [],
    }
    prosody_input = {
        "reference_text": result.reference_text,
        "selected_reference_pronunciation": result.selected_reference_candidate.pronunciation,
        "selected_reference_ipa": result.selected_reference_candidate.ipa.normalized_text,
        "gate_summary": gate_summary,
        "phoneme_segments": [
            _serialize_alignment_segment(segment)
            for segment in (result.forced_alignment_result.segments if result.forced_alignment_result is not None else [])
        ],
        "alignment_confidence": _serialize_alignment_confidence(result.alignment_confidence_report),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": timestamp,
        "profile": profile,
        "audio_source_name": audio_source_name,
        "artifact_bundle": {
            "artifact_id": artifact_id,
            "json_file_name": json_file_name,
            "audio_file_name": audio_file_name,
        },
        "status": {
            "evaluation_status": result.evaluation_status,
            "status_message": result.status_message,
        },
        "reference": {
            "text": result.reference_text,
            "representative_pronunciation": result.reference_pronunciation,
            "representative_ipa": _serialize_ipa_sequence(result.reference_ipa),
            "selected_reference_candidate": _serialize_candidate(result.selected_reference_candidate),
            "candidates": [_serialize_candidate(candidate) for candidate in result.reference_candidates],
        },
        "recognition": {
            "user_ipa_raw": result.user_ipa_raw,
            "user_ipa_normalized": result.user_ipa_normalized,
            "user_tokens": [_serialize_phone_token(token) for token in result.user_tokens],
            "raw_label_text": result.debug.get("raw_label_text"),
            "raw_labels": result.debug.get("raw_labels"),
        },
        "gates": gate_summary,
        "alignment": {
            "coarse": _serialize_alignment_result(result.alignment_result),
            "forced": _serialize_forced_alignment(result.forced_alignment_result),
        },
        "evaluation": {
            "score_breakdown": _serialize_score_breakdown(result.score_breakdown),
            "feedback_report": feedback_payload,
        },
        "llm_feedback_input": llm_feedback_input,
        "prosody_input": prosody_input,
        "debug": result.debug,
    }
    return payload


def save_evaluation_bundle(
    result: EvaluationResult,
    *,
    profile: str,
    audio_source_name: str | None = None,
    source_audio_path: str | Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path | None]:
    output_dir = out_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    artifact_dir = output_dir / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    json_file_name = f"{artifact_id}.json"
    json_path = artifact_dir / json_file_name

    audio_path: Path | None = None
    audio_file_name: str | None = None
    if source_audio_path is not None and Path(source_audio_path).exists():
        audio_suffix = _resolve_audio_suffix(audio_source_name, source_audio_path)
        audio_file_name = f"{artifact_id}{audio_suffix}"
        audio_path = artifact_dir / audio_file_name
        shutil.copy2(source_audio_path, audio_path)

    payload = build_evaluation_backdata(
        result,
        profile=profile,
        audio_source_name=audio_source_name,
        artifact_id=artifact_id,
        json_file_name=json_file_name,
        audio_file_name=audio_file_name,
    )
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_dir": artifact_dir,
        "json_path": json_path,
        "audio_path": audio_path,
    }


def save_evaluation_backdata(
    result: EvaluationResult,
    *,
    profile: str,
    audio_source_name: str | None = None,
    source_audio_path: str | Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    bundle_paths = save_evaluation_bundle(
        result,
        profile=profile,
        audio_source_name=audio_source_name,
        source_audio_path=source_audio_path,
        out_dir=out_dir,
    )
    return bundle_paths["json_path"]
