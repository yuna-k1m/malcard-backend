from __future__ import annotations

"""Audio quality and forced-alignment confidence gates."""

import math
from collections import Counter

import numpy as np

from src.confidence_calibration import calibration_debug_summary, get_token_thresholds
from src.types import AlignmentConfidenceReport, AudioQualityReport, ForcedAlignmentResult


PAUSE_BLANK_MEAN_THRESHOLD = 0.70
PAUSE_BLANK_DOMINANT_THRESHOLD = 0.65
PAUSE_BLANK_DOMINANT_RATIO_THRESHOLD = 0.60
PAUSE_MIN_EVIDENCE_FRAMES = 3
LOW_CONFIDENCE_THRESHOLD = 0.20
VERY_LOW_CONFIDENCE_THRESHOLD = 0.05


def calculate_frame_energy(audio: np.ndarray, sampling_rate: int, frame_duration: float = 0.02) -> list[float]:
    frame_size = max(1, int(sampling_rate * frame_duration))
    frames = [audio[index:index + frame_size] for index in range(0, len(audio), frame_size)] or [audio]
    return [float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0 for frame in frames]


def trim_audio_edges(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    frame_duration: float = 0.02,
    padding_sec: float = 0.15,
    threshold_floor: float = 0.01,
    threshold_ratio: float = 0.08,
) -> tuple[np.ndarray, int, int]:
    """Trim leading/trailing silence while keeping a small edge padding."""

    if audio.size == 0:
        return audio, 0, 0

    frame_size = max(1, int(sampling_rate * frame_duration))
    padding = max(1, int(sampling_rate * padding_sec))
    frame_energy = calculate_frame_energy(audio, sampling_rate, frame_duration=frame_duration)
    if not frame_energy:
        return audio, 0, len(audio)

    threshold = max(threshold_floor, threshold_ratio * max(frame_energy))
    active_frames = [index for index, energy in enumerate(frame_energy) if energy >= threshold]
    if not active_frames:
        return audio, 0, len(audio)

    start_sample = max(0, active_frames[0] * frame_size - padding)
    end_sample = min(len(audio), (active_frames[-1] + 1) * frame_size + padding)
    if end_sample <= start_sample:
        return audio, 0, len(audio)
    return audio[start_sample:end_sample], start_sample, end_sample


def analyze_audio_quality(audio: np.ndarray, sampling_rate: int) -> AudioQualityReport:
    duration_sec = len(audio) / float(sampling_rate) if sampling_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if audio.size else 1.0

    frame_energy = calculate_frame_energy(audio, sampling_rate)
    silence_ratio = float(sum(1 for energy in frame_energy if energy < 0.02) / len(frame_energy)) if frame_energy else 1.0

    reasons: list[str] = []
    if duration_sec < 0.6:
        reasons.append("음성이 너무 짧습니다.")
    if rms_db < -35.0:
        reasons.append("입력 음량이 너무 낮습니다.")
    if silence_ratio > 0.85:
        reasons.append("무음 비율이 너무 높습니다.")
    if clipping_ratio > 0.08:
        reasons.append("입력 신호에 clipping이 많습니다.")

    return AudioQualityReport(
        passed=not reasons,
        duration_sec=duration_sec,
        rms_db=rms_db,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        reasons=reasons,
    )


def _blank_posteriors(logits: np.ndarray | list[list[float]] | None, blank_id: int | None) -> np.ndarray | None:
    if logits is None or blank_id is None:
        return None

    logits_array = np.asarray(logits, dtype=np.float32)
    if logits_array.ndim != 2 or logits_array.shape[0] == 0:
        return None
    if blank_id < 0 or blank_id >= logits_array.shape[1]:
        return None

    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    denominators = np.maximum(np.sum(exp_logits, axis=1), 1e-12)
    return exp_logits[:, blank_id] / denominators


def _gap_evidence(
    gap: dict,
    *,
    blank_probs: np.ndarray | None,
    frame_confidence: np.ndarray | None,
) -> None:
    frame_start = int(gap["previous_frame_end"]) + 1
    frame_end_exclusive = int(gap["next_frame_start"])
    if frame_end_exclusive < frame_start:
        frame_end_exclusive = frame_start

    evidence_frame_count = max(0, frame_end_exclusive - frame_start)
    gap["gap_frame_start"] = frame_start
    gap["gap_frame_end_exclusive"] = frame_end_exclusive
    gap["gap_frame_count"] = evidence_frame_count
    gap["blank_mean"] = None
    gap["blank_dominant_ratio"] = None
    gap["frame_confidence_mean"] = None
    gap["frame_confidence_low_ratio"] = None
    gap["pause_like"] = False

    if evidence_frame_count <= 0:
        return

    if blank_probs is not None and frame_start < len(blank_probs):
        blank_slice = blank_probs[frame_start:min(frame_end_exclusive, len(blank_probs))]
        if blank_slice.size:
            gap["blank_mean"] = float(np.mean(blank_slice))
            gap["blank_dominant_ratio"] = float(np.mean(blank_slice >= PAUSE_BLANK_DOMINANT_THRESHOLD))

    if frame_confidence is not None and frame_start < len(frame_confidence):
        confidence_slice = frame_confidence[frame_start:min(frame_end_exclusive, len(frame_confidence))]
        if confidence_slice.size:
            gap["frame_confidence_mean"] = float(np.mean(confidence_slice))
            gap["frame_confidence_low_ratio"] = float(np.mean(confidence_slice < 0.35))

    blank_mean = gap["blank_mean"]
    blank_dominant_ratio = gap["blank_dominant_ratio"]
    gap["pause_like"] = (
        evidence_frame_count >= PAUSE_MIN_EVIDENCE_FRAMES
        and blank_mean is not None
        and blank_dominant_ratio is not None
        and blank_mean >= PAUSE_BLANK_MEAN_THRESHOLD
        and blank_dominant_ratio >= PAUSE_BLANK_DOMINANT_RATIO_THRESHOLD
    )


def _confidence_issue_summary_for_segments(
    result: ForcedAlignmentResult,
    *,
    calibration_stats: dict | None = None,
    mode: str = "raw",
) -> dict:
    segments = result.segments
    confidences = [segment.confidence for segment in segments]
    token_count = max(1, len(confidences))
    threshold_by_index = []
    source_counts: Counter[str] = Counter()
    for segment in segments:
        if mode == "calibrated":
            thresholds = get_token_thresholds(
                calibration_stats,
                segment.token,
                default_low_threshold=LOW_CONFIDENCE_THRESHOLD,
                default_very_low_threshold=VERY_LOW_CONFIDENCE_THRESHOLD,
            )
        else:
            thresholds = {
                "source": "raw_default",
                "count": None,
                "low_threshold": LOW_CONFIDENCE_THRESHOLD,
                "very_low_threshold": VERY_LOW_CONFIDENCE_THRESHOLD,
            }
        threshold_by_index.append(thresholds)
        source_counts[str(thresholds["source"])] += 1

    low_segments = [
        (index, segment)
        for index, segment in enumerate(segments)
        if segment.confidence < threshold_by_index[index]["low_threshold"]
    ]
    very_low_segments = [
        (index, segment)
        for index, segment in enumerate(segments)
        if segment.confidence < threshold_by_index[index]["very_low_threshold"]
    ]
    very_low_count = len(very_low_segments)
    low_count = len(low_segments)
    very_low_ratio = very_low_count / token_count
    low_ratio = low_count / token_count
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    symbol_counts = Counter(segment.token for _, segment in very_low_segments)
    dominant_symbols = []
    for symbol, count in symbol_counts.most_common(8):
        symbol_confidences = [
            segment.confidence
            for _, segment in very_low_segments
            if segment.token == symbol
        ]
        symbol_thresholds = [
            threshold_by_index[index]["very_low_threshold"]
            for index, segment in very_low_segments
            if segment.token == symbol
        ]
        dominant_symbols.append(
            {
                "token": symbol,
                "count": count,
                "ratio_among_very_low": count / max(1, very_low_count),
                "min_confidence": min(symbol_confidences) if symbol_confidences else None,
                "mean_confidence": float(np.mean(symbol_confidences)) if symbol_confidences else None,
                "mean_very_low_threshold": float(np.mean(symbol_thresholds)) if symbol_thresholds else None,
            }
        )

    very_low_runs = []
    current_run = []
    very_low_indices = {index for index, _ in very_low_segments}
    for index, segment in enumerate(segments):
        if index in very_low_indices:
            current_run.append((index, segment))
            continue
        if current_run:
            very_low_runs.append(current_run)
            current_run = []
    if current_run:
        very_low_runs.append(current_run)

    serialized_runs = [
        {
            "start_index": run[0][0],
            "end_index": run[-1][0],
            "length": len(run),
            "tokens": [segment.token for _, segment in run],
            "min_confidence": min(segment.confidence for _, segment in run),
        }
        for run in very_low_runs
    ]
    serialized_runs.sort(key=lambda item: (item["length"], -item["min_confidence"]), reverse=True)

    worst_segments = sorted(
        [
            {
                "index": index,
                "token": segment.token,
                "label": segment.label,
                "confidence": segment.confidence,
                "low_threshold": threshold_by_index[index]["low_threshold"],
                "very_low_threshold": threshold_by_index[index]["very_low_threshold"],
                "threshold_source": threshold_by_index[index]["source"],
                "start_time": segment.start_time,
                "end_time": segment.end_time,
            }
            for index, segment in enumerate(segments)
        ],
        key=lambda item: item["confidence"],
    )[:10]

    unique_very_low_symbols = len(symbol_counts)
    dominant_ratio = dominant_symbols[0]["ratio_among_very_low"] if dominant_symbols else 0.0
    unique_ratio = unique_very_low_symbols / max(1, very_low_count)
    if very_low_count == 0 and low_ratio <= 0.15:
        scope = "none"
    elif (
        very_low_ratio > 0.30
        and (low_ratio > 0.45 or avg_confidence < 0.60 or unique_ratio > 0.55)
    ):
        scope = "global"
    elif very_low_count > 0 and (dominant_ratio >= 0.45 or unique_very_low_symbols <= 2):
        scope = "localized"
    elif very_low_ratio > 0.15 or low_ratio > 0.30:
        scope = "mixed"
    else:
        scope = "minor"

    return {
        "scope": scope,
        "mode": mode,
        "threshold_source_counts": dict(source_counts),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "very_low_confidence_threshold": VERY_LOW_CONFIDENCE_THRESHOLD,
        "token_count": len(segments),
        "low_confidence_count": low_count,
        "very_low_confidence_count": very_low_count,
        "low_confidence_ratio": low_ratio,
        "very_low_confidence_ratio": very_low_ratio,
        "avg_token_confidence": avg_confidence,
        "unique_very_low_token_count": unique_very_low_symbols,
        "dominant_very_low_tokens": dominant_symbols,
        "very_low_runs": serialized_runs[:8],
        "worst_segments": worst_segments,
    }


def _confidence_issue_summary(
    result: ForcedAlignmentResult,
    *,
    calibration_stats: dict | None = None,
) -> dict:
    raw_summary = _confidence_issue_summary_for_segments(result, mode="raw")
    if not calibration_stats:
        raw_summary["calibration_enabled"] = False
        raw_summary["effective_scope"] = raw_summary["scope"]
        raw_summary["effective_low_confidence_count"] = raw_summary["low_confidence_count"]
        raw_summary["effective_very_low_confidence_count"] = raw_summary["very_low_confidence_count"]
        raw_summary["effective_low_confidence_ratio"] = raw_summary["low_confidence_ratio"]
        raw_summary["effective_very_low_confidence_ratio"] = raw_summary["very_low_confidence_ratio"]
        return raw_summary

    calibrated_summary = _confidence_issue_summary_for_segments(
        result,
        calibration_stats=calibration_stats,
        mode="calibrated",
    )
    raw_summary["calibration_enabled"] = True
    raw_summary["calibration"] = calibration_debug_summary(calibration_stats)
    raw_summary["calibrated"] = calibrated_summary
    raw_summary["effective_scope"] = calibrated_summary["scope"]
    raw_summary["effective_low_confidence_count"] = calibrated_summary["low_confidence_count"]
    raw_summary["effective_very_low_confidence_count"] = calibrated_summary["very_low_confidence_count"]
    raw_summary["effective_low_confidence_ratio"] = calibrated_summary["low_confidence_ratio"]
    raw_summary["effective_very_low_confidence_ratio"] = calibrated_summary["very_low_confidence_ratio"]
    return raw_summary


def summarize_alignment_timing(
    result: ForcedAlignmentResult,
    *,
    frame_confidence: list[float] | np.ndarray | None = None,
    logits: np.ndarray | list[list[float]] | None = None,
    blank_id: int | None = None,
    calibration_stats: dict | None = None,
) -> dict:
    segments = result.segments
    durations = [max(0.0, segment.end_time - segment.start_time) for segment in segments]
    confidences = [segment.confidence for segment in segments]
    blank_probs = _blank_posteriors(logits, blank_id)
    frame_confidence_array = (
        np.asarray(frame_confidence, dtype=np.float32)
        if frame_confidence is not None
        else None
    )
    gaps = [
        {
            "previous_index": index - 1,
            "next_index": index,
            "gap": max(0.0, segments[index].start_time - segments[index - 1].end_time),
            "previous_token": segments[index - 1].token,
            "next_token": segments[index].token,
            "previous_end_time": segments[index - 1].end_time,
            "next_start_time": segments[index].start_time,
            "previous_frame_end": segments[index - 1].frame_end,
            "next_frame_start": segments[index].frame_start,
            "previous_confidence": segments[index - 1].confidence,
            "next_confidence": segments[index].confidence,
            "is_tail_gap": index >= len(segments) - 2,
        }
        for index in range(1, len(segments))
    ]
    for gap in gaps:
        _gap_evidence(
            gap,
            blank_probs=blank_probs,
            frame_confidence=frame_confidence_array,
        )

    gap_values = [gap["gap"] for gap in gaps]
    tail_gaps = gap_values[-2:] if len(gap_values) >= 2 else gap_values
    internal_gap_values = gap_values[:-2] if len(gap_values) > 2 else []

    large_gap_threshold = 0.30
    warning_gap_threshold = 0.80
    failure_internal_gap_threshold = 1.20
    very_low_confidence_threshold = VERY_LOW_CONFIDENCE_THRESHOLD

    max_gap = max(gap_values) if gap_values else 0.0
    max_tail_gap = max(tail_gaps) if tail_gaps else 0.0
    max_internal_gap = max(internal_gap_values) if internal_gap_values else 0.0
    low_confidence_count = sum(conf < LOW_CONFIDENCE_THRESHOLD for conf in confidences)
    very_low_confidence_count = sum(conf < very_low_confidence_threshold for conf in confidences)
    token_count = max(1, len(confidences))
    confidence_issue_summary = _confidence_issue_summary(
        result,
        calibration_stats=calibration_stats,
    )
    effective_very_low_confidence_ratio = confidence_issue_summary["effective_very_low_confidence_ratio"]
    effective_low_confidence_ratio = confidence_issue_summary["effective_low_confidence_ratio"]
    effective_very_low_confidence_count = confidence_issue_summary["effective_very_low_confidence_count"]
    effective_low_confidence_count = confidence_issue_summary["effective_low_confidence_count"]

    large_gaps = [gap for gap in gaps if gap["gap"] > large_gap_threshold]
    severe_gaps = [gap for gap in gaps if gap["gap"] > warning_gap_threshold]
    pause_gaps = [gap for gap in large_gaps if gap["pause_like"] and not gap["is_tail_gap"]]
    unexplained_large_gaps = [
        gap for gap in large_gaps
        if not gap["pause_like"] and not gap["is_tail_gap"]
    ]
    failure_gaps = [
        gap for gap in gaps
        if gap["gap"] > failure_internal_gap_threshold
        and not gap["pause_like"]
        and not gap["is_tail_gap"]
    ]
    max_unexplained_internal_gap = max([gap["gap"] for gap in failure_gaps], default=0.0)
    max_pause_gap = max([gap["gap"] for gap in pause_gaps], default=0.0)

    if result.coverage < 0.90 or max_tail_gap > large_gap_threshold or max_unexplained_internal_gap > failure_internal_gap_threshold:
        quality_level = "discardable"
    elif severe_gaps or len(large_gaps) >= 3 or effective_very_low_confidence_ratio > 0.15:
        quality_level = "usable_with_warnings"
    elif large_gaps or effective_very_low_confidence_ratio > 0.10:
        quality_level = "minor_warnings"
    else:
        quality_level = "clean"

    top_gaps = sorted(gaps, key=lambda gap: gap["gap"], reverse=True)[:10]
    return {
        "quality_level": quality_level,
        "large_gap_threshold": large_gap_threshold,
        "warning_gap_threshold": warning_gap_threshold,
        "failure_internal_gap_threshold": failure_internal_gap_threshold,
        "segment_count": len(segments),
        "max_gap": max_gap,
        "max_internal_gap": max_internal_gap,
        "max_unexplained_internal_gap": max_unexplained_internal_gap,
        "max_pause_gap": max_pause_gap,
        "max_tail_gap": max_tail_gap,
        "large_gap_count": len(large_gaps),
        "warning_gap_count": len(severe_gaps),
        "failure_gap_count": len(failure_gaps),
        "pause_gap_count": len(pause_gaps),
        "unexplained_large_gap_count": len(unexplained_large_gaps),
        "median_duration": float(np.median(durations)) if durations else 0.0,
        "max_duration": max(durations) if durations else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "low_confidence_ratio": low_confidence_count / token_count,
        "very_low_confidence_ratio": very_low_confidence_count / token_count,
        "low_confidence_count": low_confidence_count,
        "very_low_confidence_count": very_low_confidence_count,
        "effective_low_confidence_ratio": effective_low_confidence_ratio,
        "effective_very_low_confidence_ratio": effective_very_low_confidence_ratio,
        "effective_low_confidence_count": effective_low_confidence_count,
        "effective_very_low_confidence_count": effective_very_low_confidence_count,
        "calibrated_low_confidence_ratio": (
            (confidence_issue_summary.get("calibrated") or {}).get("low_confidence_ratio")
            if confidence_issue_summary.get("calibration_enabled")
            else None
        ),
        "calibrated_very_low_confidence_ratio": (
            (confidence_issue_summary.get("calibrated") or {}).get("very_low_confidence_ratio")
            if confidence_issue_summary.get("calibration_enabled")
            else None
        ),
        "confidence_calibration": calibration_debug_summary(calibration_stats),
        "confidence_issue_summary": confidence_issue_summary,
        "top_gaps": top_gaps,
    }


def decide_prosody_alignment_usage(
    timing_summary: dict | None,
    *,
    evaluation_status: str,
    alignment_gate_passed: bool | None,
) -> dict:
    reasons: list[str] = []
    quality_level = (timing_summary or {}).get("quality_level")
    warning_gap_count = int((timing_summary or {}).get("warning_gap_count") or 0)
    large_gap_count = int((timing_summary or {}).get("large_gap_count") or 0)
    pause_gap_count = int((timing_summary or {}).get("pause_gap_count") or 0)
    unexplained_large_gap_count = int((timing_summary or {}).get("unexplained_large_gap_count") or 0)
    max_tail_gap = float((timing_summary or {}).get("max_tail_gap") or 0.0)
    large_gap_threshold = float((timing_summary or {}).get("large_gap_threshold") or 0.30)
    very_low_confidence_ratio = float(
        (timing_summary or {}).get(
            "effective_very_low_confidence_ratio",
            (timing_summary or {}).get("very_low_confidence_ratio") or 0.0,
        )
        or 0.0
    )
    confidence_issue_summary = (timing_summary or {}).get("confidence_issue_summary") or {}
    confidence_scope = confidence_issue_summary.get("effective_scope") or confidence_issue_summary.get("scope")
    raw_confidence_scope = confidence_issue_summary.get("scope")
    calibrated_confidence_scope = (confidence_issue_summary.get("calibrated") or {}).get("scope")
    confidence_limited = (
        very_low_confidence_ratio > 0.15
        or (confidence_scope in {"global", "mixed"} and very_low_confidence_ratio > 0.0)
    )
    minor_confidence_limited = (
        very_low_confidence_ratio > 0.10
        or (confidence_scope in {"global", "mixed"} and very_low_confidence_ratio > 0.0)
    )
    pause_limited = pause_gap_count > 0
    gap_limited = warning_gap_count > 0 or large_gap_count >= 3 or unexplained_large_gap_count > 0
    tail_limited = max_tail_gap > large_gap_threshold
    alignment_gate_limited = alignment_gate_passed is False

    limitation_causes = {
        "alignment_gate": alignment_gate_limited,
        "pause": pause_limited,
        "confidence": confidence_limited or minor_confidence_limited,
        "gap": gap_limited,
        "tail": tail_limited,
    }
    limited_by = [
        name
        for name, enabled in limitation_causes.items()
        if enabled
    ]

    def payload(
        *,
        alignment_for_prosody: bool,
        detailed_timing_allowed: bool,
        recommended_usage: str,
        reasons: list[str],
    ) -> dict:
        return {
            "alignment_for_prosody": alignment_for_prosody,
            "detailed_timing_allowed": detailed_timing_allowed,
            "recommended_usage": recommended_usage,
            "alignment_quality_level": quality_level,
            "reasons": reasons,
            "limited_by": limited_by,
            "limitation_causes": limitation_causes,
            "confidence_issue_scope": confidence_scope,
            "raw_confidence_issue_scope": raw_confidence_scope,
            "calibrated_confidence_issue_scope": calibrated_confidence_scope,
            "raw_very_low_confidence_ratio": (timing_summary or {}).get("very_low_confidence_ratio"),
            "calibrated_very_low_confidence_ratio": (timing_summary or {}).get("calibrated_very_low_confidence_ratio"),
            "effective_very_low_confidence_ratio": very_low_confidence_ratio,
        }

    if not timing_summary:
        return payload(
            alignment_for_prosody=False,
            detailed_timing_allowed=False,
            recommended_usage="disabled",
            reasons=["forced_alignment_missing"],
        )

    if evaluation_status != "ready":
        reasons.append("evaluation_not_ready")
        return payload(
            alignment_for_prosody=False,
            detailed_timing_allowed=False,
            recommended_usage="disabled",
            reasons=reasons,
        )

    if quality_level == "discardable":
        reasons.append("timing_quality_discardable")
        if alignment_gate_limited:
            reasons.append("alignment_gate_not_ready")
        if confidence_limited:
            reasons.append(f"confidence_limited_{confidence_scope or 'unknown'}")
        if pause_limited:
            reasons.append("pause_gap_detected")
        if tail_limited:
            reasons.append("tail_gap_detected")
        return payload(
            alignment_for_prosody=True,
            detailed_timing_allowed=True,
            recommended_usage="limited",
            reasons=reasons,
        )

    if quality_level == "usable_with_warnings":
        if alignment_gate_limited:
            reasons.append("alignment_gate_not_ready")
        if pause_gap_count:
            reasons.append("pause_gap_detected")
        if warning_gap_count:
            reasons.append("large_internal_gap_warning")
        if large_gap_count >= 3:
            reasons.append("many_large_gaps")
        if very_low_confidence_ratio > 0.15:
            reasons.append("many_very_low_confidence_segments")
        if confidence_limited:
            reasons.append(f"confidence_limited_{confidence_scope or 'unknown'}")
        return payload(
            alignment_for_prosody=True,
            detailed_timing_allowed=True,
            recommended_usage="limited",
            reasons=reasons or ["timing_quality_usable_with_warnings"],
        )

    if quality_level == "minor_warnings":
        if alignment_gate_limited:
            reasons.append("alignment_gate_not_ready")
        if pause_gap_count:
            reasons.append("pause_gap_detected")
        if large_gap_count:
            reasons.append("minor_gap_warning")
        if very_low_confidence_ratio > 0.10:
            reasons.append("minor_low_confidence_warning")
        if minor_confidence_limited:
            reasons.append(f"confidence_limited_{confidence_scope or 'minor'}")
        return payload(
            alignment_for_prosody=True,
            detailed_timing_allowed=True,
            recommended_usage="cautious",
            reasons=reasons or ["minor_timing_warning"],
        )

    if alignment_gate_limited:
        return payload(
            alignment_for_prosody=True,
            detailed_timing_allowed=True,
            recommended_usage="limited",
            reasons=["alignment_gate_not_ready"],
        )

    return payload(
        alignment_for_prosody=True,
        detailed_timing_allowed=True,
        recommended_usage="full",
        reasons=[],
    )


def assess_alignment_confidence(
    result: ForcedAlignmentResult,
    *,
    frame_confidence: list[float] | np.ndarray | None = None,
    logits: np.ndarray | list[list[float]] | None = None,
    blank_id: int | None = None,
    calibration_stats: dict | None = None,
) -> AlignmentConfidenceReport:
    """
    Check whether frame-level forced alignment is reliable enough.

    Policy:
    - Do not fail only because one token has extremely low confidence.
    - Use ratios instead of a single minimum value.
    - Keep thresholds realistic for CTC forced alignment.
    """

    reasons: list[str] = []

    confidences = [segment.confidence for segment in result.segments]
    token_count = max(1, len(confidences))

    min_token_confidence = min(confidences) if confidences else 0.0
    low_conf_count = sum(conf < LOW_CONFIDENCE_THRESHOLD for conf in confidences)
    very_low_conf_count = sum(conf < VERY_LOW_CONFIDENCE_THRESHOLD for conf in confidences)

    low_conf_ratio = low_conf_count / token_count
    very_low_conf_ratio = very_low_conf_count / token_count

    debug_notes: list[str] = []
    timing_summary = summarize_alignment_timing(
        result,
        frame_confidence=frame_confidence,
        logits=logits,
        blank_id=blank_id,
        calibration_stats=calibration_stats,
    )
    median_duration = timing_summary["median_duration"]
    max_duration = timing_summary["max_duration"]
    max_gap = timing_summary["max_gap"]
    max_tail_gap = timing_summary["max_tail_gap"]
    max_unexplained_internal_gap = timing_summary["max_unexplained_internal_gap"]
    pause_gap_count = timing_summary["pause_gap_count"]
    confidence_issue_summary = timing_summary["confidence_issue_summary"]
    effective_low_conf_ratio = float(timing_summary.get("effective_low_confidence_ratio", low_conf_ratio) or 0.0)
    effective_very_low_conf_ratio = float(
        timing_summary.get("effective_very_low_confidence_ratio", very_low_conf_ratio) or 0.0
    )
    effective_confidence_scope = confidence_issue_summary.get("effective_scope") or confidence_issue_summary["scope"]
    final_confidences = confidences[-2:]
    final_segments = result.segments[-2:]

    if result.coverage < 0.90:
        reasons.append(
            f"정답 음소 대부분이 시간축에 안정적으로 배치되지 않았습니다. "
            f"coverage={result.coverage:.3f}"
        )

    if result.avg_token_confidence < 0.45:
        reasons.append(
            f"정렬 경로의 평균 음소 신뢰도가 낮습니다. "
            f"avg_token_confidence={result.avg_token_confidence:.3f}"
        )

    if low_conf_ratio - effective_low_conf_ratio > 0.05:
        debug_notes.append(
            f"calibrated_low_conf_ratio={effective_low_conf_ratio:.3f}, raw_low_conf_ratio={low_conf_ratio:.3f}"
        )

    if very_low_conf_ratio - effective_very_low_conf_ratio > 0.05:
        debug_notes.append(
            "calibrated_very_low_conf_ratio="
            f"{effective_very_low_conf_ratio:.3f}, raw_very_low_conf_ratio={very_low_conf_ratio:.3f}"
        )

    if effective_low_conf_ratio > 0.45:
        reasons.append(
            f"신뢰도 0.20 미만 음소 비율이 높습니다. "
            f"low_conf_ratio={effective_low_conf_ratio:.3f}"
        )

    if effective_very_low_conf_ratio > 0.30:
        reasons.append(
            f"신뢰도 0.05 미만 음소 비율이 높습니다. "
            f"very_low_conf_ratio={effective_very_low_conf_ratio:.3f}, "
            f"scope={effective_confidence_scope}"
        )
    elif effective_very_low_conf_ratio > 0.15:
        debug_notes.append(
            f"very_low_confidence_scope={effective_confidence_scope}"
        )

    # Warning only, not a hard failure.
    if len(confidences) >= 10 and min_token_confidence < 0.005:
        debug_notes.append(
            f"min_token_confidence={min_token_confidence:.6f}"
        )

    if result.normalized_log_prob < -1.50:
        reasons.append(
            f"전체 정렬 경로의 로그확률이 낮습니다. "
            f"normalized_log_prob={result.normalized_log_prob:.3f}"
        )

    if result.blank_ratio > 0.97:
        reasons.append(
            f"blank frame 비율이 너무 높습니다. "
            f"blank_ratio={result.blank_ratio:.3f}"
        )

    final_very_low_flags = []
    for segment, confidence in zip(final_segments, final_confidences):
        thresholds = get_token_thresholds(
            calibration_stats,
            segment.token,
            default_low_threshold=LOW_CONFIDENCE_THRESHOLD,
            default_very_low_threshold=VERY_LOW_CONFIDENCE_THRESHOLD,
        )
        final_very_low_flags.append(confidence < thresholds["very_low_threshold"])

    if len(final_confidences) == 2 and all(final_very_low_flags):
        reasons.append(
            "Forced alignment confidence is too low on the final phones. "
            f"final_confidences={[round(conf, 6) for conf in final_confidences]}"
        )

    if median_duration > 0 and max_duration > max(0.75, median_duration * 8.0):
        reasons.append(
            "Forced alignment timing has an abnormally long phone interval. "
            f"max_duration={max_duration:.3f}, median_duration={median_duration:.3f}"
        )

    if max_tail_gap > 0.30:
        reasons.append(
            "Forced alignment timing has an abnormal gap near the final phones. "
            f"max_tail_gap={max_tail_gap:.3f}"
        )
    elif max_unexplained_internal_gap > 1.20:
        reasons.append(
            "Forced alignment timing has an abnormal internal phone gap. "
            f"max_gap={max_unexplained_internal_gap:.3f}"
        )
    elif pause_gap_count and max_gap > 1.20:
        debug_notes.append(
            "internal_pause_gap_detected="
            f"{pause_gap_count}, max_pause_gap={timing_summary['max_pause_gap']:.3f}"
        )

    if reasons:
        message = " ".join(reasons)
    else:
        message = "forced alignment를 신뢰할 수 있습니다."

    if debug_notes:
        message = message + " debug: " + ", ".join(debug_notes)

    return AlignmentConfidenceReport(
        passed=not reasons,
        avg_token_confidence=result.avg_token_confidence,
        coverage=result.coverage,
        normalized_log_prob=result.normalized_log_prob,
        message=message,
    )
