from __future__ import annotations

"""Frame-level CTC forced alignment for reference IPA tokens."""

import numpy as np

from src.label_to_ipa import ipa_tokens_to_labels
from src.types import AlignmentSegment, ForcedAlignmentResult, PronunciationCandidate


MAX_BRIDGED_BLANK_FRAMES = 20
MAX_EDGE_PADDING_FRAMES = 2
PAUSE_BLANK_PROB_THRESHOLD = 0.85
PAUSE_FRAME_CONFIDENCE_THRESHOLD = 0.70
PAUSE_MIN_COMPRESS_FRAMES = 8
PAUSE_KEEP_EDGE_FRAMES = 2
PAUSE_ENERGY_ABSOLUTE_THRESHOLD = 0.01
PAUSE_ENERGY_RELATIVE_THRESHOLD = 0.15
WEAK_EVIDENCE_TOKENS = {
    "t̚",
    "k̚",
    "p̚",
    "ŋ",
    "t",
    "k",
    "p",
    "jo",
    "l",
}


def _build_extended_sequence(target_ids: list[int], blank_id: int) -> list[int]:
    sequence = [blank_id]
    for token_id in target_ids:
        sequence.append(token_id)
        sequence.append(blank_id)
    return sequence


def _viterbi_ctc_path(log_probs: np.ndarray, target_ids: list[int], blank_id: int) -> tuple[list[int], float]:
    extended = _build_extended_sequence(target_ids, blank_id)
    num_frames = log_probs.shape[0]
    num_states = len(extended)

    back_dtype = np.int16 if num_states <= np.iinfo(np.int16).max else np.int32
    back = np.full((num_frames, num_states), -1, dtype=back_dtype)
    previous_scores = np.full(num_states, -np.inf, dtype=np.float32)
    current_scores = np.full(num_states, -np.inf, dtype=np.float32)

    previous_scores[0] = float(log_probs[0, blank_id])
    if num_states > 1:
        previous_scores[1] = float(log_probs[0, extended[1]])

    for frame in range(1, num_frames):
        current_scores.fill(-np.inf)
        for state in range(num_states):
            best_score = previous_scores[state]
            best_prev = state
            if state - 1 >= 0:
                previous_score = previous_scores[state - 1]
                if previous_score > best_score:
                    best_score = previous_score
                    best_prev = state - 1
            if (
                state - 2 >= 0
                and extended[state] != blank_id
                and extended[state] != extended[state - 2]
            ):
                skip_score = previous_scores[state - 2]
                if skip_score > best_score:
                    best_score = skip_score
                    best_prev = state - 2

            current_scores[state] = best_score + float(log_probs[frame, extended[state]])
            back[frame, state] = best_prev
        previous_scores, current_scores = current_scores, previous_scores

    best_score = previous_scores[num_states - 1]
    best_state = num_states - 1
    if num_states > 1 and previous_scores[num_states - 2] > best_score:
        best_score = previous_scores[num_states - 2]
        best_state = num_states - 2

    states = [best_state]
    state = best_state
    for frame in range(num_frames - 1, 0, -1):
        state = int(back[frame, state])
        states.append(state)
    states.reverse()
    return states, float(best_score)


def _logaddexp3(first: float, second: float, third: float) -> float:
    return float(np.logaddexp(np.logaddexp(first, second), third))


def _forward_backward_state_posteriors(
    log_probs: np.ndarray,
    target_ids: list[int],
    blank_id: int,
) -> tuple[np.ndarray | None, float | None]:
    extended = _build_extended_sequence(target_ids, blank_id)
    num_frames = log_probs.shape[0]
    num_states = len(extended)
    if num_frames == 0 or num_states == 0:
        return None, None

    alpha = np.full((num_frames, num_states), -np.inf, dtype=np.float32)
    beta = np.full((num_frames, num_states), -np.inf, dtype=np.float32)
    alpha[0, 0] = float(log_probs[0, blank_id])
    if num_states > 1:
        alpha[0, 1] = float(log_probs[0, extended[1]])

    for frame in range(1, num_frames):
        for state in range(num_states):
            stay = float(alpha[frame - 1, state])
            previous = float(alpha[frame - 1, state - 1]) if state - 1 >= 0 else -np.inf
            skip = -np.inf
            if (
                state - 2 >= 0
                and extended[state] != blank_id
                and extended[state] != extended[state - 2]
            ):
                skip = float(alpha[frame - 1, state - 2])
            alpha[frame, state] = _logaddexp3(stay, previous, skip) + float(log_probs[frame, extended[state]])

    beta[num_frames - 1, num_states - 1] = 0.0
    if num_states > 1:
        beta[num_frames - 1, num_states - 2] = 0.0

    for frame in range(num_frames - 2, -1, -1):
        for state in range(num_states):
            stay = float(log_probs[frame + 1, extended[state]]) + float(beta[frame + 1, state])
            next_state = (
                float(log_probs[frame + 1, extended[state + 1]]) + float(beta[frame + 1, state + 1])
                if state + 1 < num_states
                else -np.inf
            )
            skip = -np.inf
            if (
                state + 2 < num_states
                and extended[state + 2] != blank_id
                and extended[state + 2] != extended[state]
            ):
                skip = float(log_probs[frame + 1, extended[state + 2]]) + float(beta[frame + 1, state + 2])
            beta[frame, state] = _logaddexp3(stay, next_state, skip)

    total_log_prob = float(np.logaddexp(alpha[-1, -1], alpha[-1, -2] if num_states > 1 else -np.inf))
    if not np.isfinite(total_log_prob):
        return None, None
    return alpha + beta - total_log_prob, total_log_prob


def _energy_threshold(frame_energy: np.ndarray | None) -> float | None:
    if frame_energy is None or frame_energy.size == 0:
        return None
    positive = frame_energy[frame_energy > 0.0]
    if positive.size == 0:
        return PAUSE_ENERGY_ABSOLUTE_THRESHOLD
    high_energy = float(np.percentile(positive, 95))
    return max(PAUSE_ENERGY_ABSOLUTE_THRESHOLD, high_energy * PAUSE_ENERGY_RELATIVE_THRESHOLD)


def _pause_candidate_mask(
    log_probs: np.ndarray,
    blank_id: int,
    *,
    frame_confidence: list[float] | np.ndarray | None = None,
    frame_energy: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    blank_probs = np.exp(log_probs[:, blank_id])
    if frame_confidence is None:
        confidence_array = np.exp(np.max(log_probs, axis=1))
    else:
        confidence_array = np.asarray(frame_confidence, dtype=np.float32)
        if confidence_array.shape[0] != log_probs.shape[0]:
            confidence_array = np.exp(np.max(log_probs, axis=1))

    energy_array = (
        np.asarray(frame_energy, dtype=np.float32)
        if frame_energy is not None
        else None
    )
    energy_threshold = _energy_threshold(energy_array)
    energy_mask = np.ones(log_probs.shape[0], dtype=bool)
    if energy_array is not None and energy_array.shape[0] == log_probs.shape[0] and energy_threshold is not None:
        energy_mask = energy_array <= energy_threshold

    mask = (
        (blank_probs >= PAUSE_BLANK_PROB_THRESHOLD)
        & (confidence_array >= PAUSE_FRAME_CONFIDENCE_THRESHOLD)
        & energy_mask
    )
    return mask, {
        "blank_prob_threshold": PAUSE_BLANK_PROB_THRESHOLD,
        "frame_confidence_threshold": PAUSE_FRAME_CONFIDENCE_THRESHOLD,
        "energy_threshold": energy_threshold,
        "energy_available": energy_array is not None and energy_array.shape[0] == log_probs.shape[0],
    }


def _compress_pause_frames(
    log_probs: np.ndarray,
    blank_id: int,
    *,
    min_remaining_frames: int,
    frame_confidence: list[float] | np.ndarray | None = None,
    frame_energy: list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    num_frames = log_probs.shape[0]
    frame_map = np.arange(num_frames, dtype=np.int32)
    pause_mask, debug = _pause_candidate_mask(
        log_probs,
        blank_id,
        frame_confidence=frame_confidence,
        frame_energy=frame_energy,
    )
    remove_mask = np.zeros(num_frames, dtype=bool)
    runs = []
    index = 0
    while index < num_frames:
        if not pause_mask[index]:
            index += 1
            continue
        start = index
        while index < num_frames and pause_mask[index]:
            index += 1
        end = index
        run_length = end - start
        if run_length < PAUSE_MIN_COMPRESS_FRAMES:
            continue
        remove_start = min(end, start + PAUSE_KEEP_EDGE_FRAMES)
        remove_end = max(remove_start, end - PAUSE_KEEP_EDGE_FRAMES)
        if remove_end <= remove_start:
            continue
        remove_mask[remove_start:remove_end] = True
        runs.append(
            {
                "start_frame": start,
                "end_frame_exclusive": end,
                "length": run_length,
                "removed_start_frame": remove_start,
                "removed_end_frame_exclusive": remove_end,
                "removed_length": remove_end - remove_start,
            }
        )

    keep_indices = np.flatnonzero(~remove_mask).astype(np.int32)
    removed_count = int(remove_mask.sum())
    debug.update(
        {
            "enabled": bool(removed_count),
            "min_compress_frames": PAUSE_MIN_COMPRESS_FRAMES,
            "keep_edge_frames": PAUSE_KEEP_EDGE_FRAMES,
            "candidate_frame_count": int(pause_mask.sum()),
            "removed_frame_count": removed_count,
            "original_frame_count": num_frames,
            "compressed_frame_count": int(keep_indices.size),
            "runs": runs[:12],
        }
    )
    if removed_count == 0:
        return log_probs, frame_map, debug
    if keep_indices.size < max(1, min_remaining_frames):
        debug["enabled"] = False
        debug["skipped_reason"] = "not_enough_frames_after_pause_compression"
        return log_probs, frame_map, debug
    return log_probs[keep_indices], keep_indices, debug


def _frame_boundaries(frame_timestamps: list[float], num_frames: int) -> list[float]:
    if num_frames <= 0:
        return [0.0]

    if len(frame_timestamps) >= num_frames + 1:
        return [float(timestamp) for timestamp in frame_timestamps[:num_frames + 1]]

    if len(frame_timestamps) == num_frames and num_frames > 1:
        audio_duration = float(frame_timestamps[-1])
        if audio_duration > 0.0:
            return [audio_duration * index / num_frames for index in range(num_frames + 1)]

    if frame_timestamps:
        frame_width = (
            float(frame_timestamps[1] - frame_timestamps[0])
            if len(frame_timestamps) > 1
            else 0.02
        )
        boundaries = [float(timestamp) for timestamp in frame_timestamps]
        while len(boundaries) < num_frames + 1:
            boundaries.append(boundaries[-1] + frame_width)
        return boundaries

    return [float(index) for index in range(num_frames + 1)]


def _segment_boundaries(label_frame_buckets: list[list[int]], num_frames: int) -> list[tuple[int, int] | None]:
    evidence_ranges: list[tuple[int, int] | None] = [
        (frame_indices[0], frame_indices[-1])
        if frame_indices else None
        for frame_indices in label_frame_buckets
    ]
    present = [index for index, frame_range in enumerate(evidence_ranges) if frame_range is not None]
    boundaries: list[tuple[int, int] | None] = [None for _ in label_frame_buckets]

    for label_index in present:
        frame_range = evidence_ranges[label_index]
        assert frame_range is not None
        raw_start, raw_end = frame_range
        start_boundary = max(0, raw_start - MAX_EDGE_PADDING_FRAMES)
        end_boundary = min(num_frames, raw_end + 1 + MAX_EDGE_PADDING_FRAMES)
        boundaries[label_index] = (start_boundary, max(start_boundary + 1, end_boundary))

    for previous_index, next_index in zip(present, present[1:]):
        previous_range = evidence_ranges[previous_index]
        next_range = evidence_ranges[next_index]
        previous_boundary = boundaries[previous_index]
        next_boundary = boundaries[next_index]
        assert previous_range is not None and next_range is not None
        assert previous_boundary is not None and next_boundary is not None

        previous_end_exclusive = previous_range[1] + 1
        next_start = next_range[0]
        blank_gap = next_start - previous_end_exclusive
        if blank_gap <= MAX_BRIDGED_BLANK_FRAMES:
            split = previous_end_exclusive + max(0, blank_gap) // 2
            boundaries[previous_index] = (previous_boundary[0], max(previous_boundary[0] + 1, split))
            boundaries[next_index] = (min(split, next_boundary[1] - 1), next_boundary[1])
        else:
            previous_end = min(previous_boundary[1], previous_end_exclusive + MAX_EDGE_PADDING_FRAMES)
            next_start_boundary = max(next_boundary[0], next_start - MAX_EDGE_PADDING_FRAMES)
            boundaries[previous_index] = (previous_boundary[0], max(previous_boundary[0] + 1, previous_end))
            boundaries[next_index] = (min(next_start_boundary, next_boundary[1] - 1), next_boundary[1])

    return boundaries


def _topk_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, min(k, values.size))
    return float(np.mean(np.partition(values, -k)[-k:]))


def _label_confidence(
    *,
    label_index: int,
    label_id: int,
    token_symbol: str,
    compressed_frame_indices: list[int],
    log_probs: np.ndarray,
    state_log_posteriors: np.ndarray | None,
) -> tuple[float, dict]:
    if not compressed_frame_indices:
        return 0.0, {
            "path_mean": 0.0,
            "posterior_topk": None,
            "posterior_peak": None,
            "method": "missing_path_evidence",
        }

    frame_indices = np.asarray(compressed_frame_indices, dtype=np.int32)
    path_probs = np.exp(log_probs[frame_indices, label_id])
    path_mean = float(np.mean(path_probs))
    path_peak = float(np.max(path_probs))

    posterior_topk = None
    posterior_peak = None
    confidence = path_mean
    method = "viterbi_emission_mean"
    if state_log_posteriors is not None:
        label_state = label_index * 2 + 1
        window_start = max(0, int(frame_indices[0]) - 2)
        window_end = min(log_probs.shape[0], int(frame_indices[-1]) + 3)
        posterior_values = np.exp(state_log_posteriors[window_start:window_end, label_state])
        posterior_topk = _topk_mean(posterior_values, 3)
        posterior_peak = float(np.max(posterior_values)) if posterior_values.size else 0.0
        short_or_weak = token_symbol in WEAK_EVIDENCE_TOKENS or len(frame_indices) <= 2
        if short_or_weak:
            confidence = max(path_mean, posterior_topk, posterior_peak * 0.85)
            method = "forward_backward_peak_topk"
        else:
            confidence = max(path_mean, (path_mean * 0.75) + (posterior_topk * 0.25))
            method = "forward_backward_blended"

    return float(np.clip(confidence, 0.0, 1.0)), {
        "path_mean": path_mean,
        "path_peak": path_peak,
        "posterior_topk": posterior_topk,
        "posterior_peak": posterior_peak,
        "method": method,
    }


def force_align_candidate(
    candidate: PronunciationCandidate,
    logits: np.ndarray | list[list[float]],
    frame_timestamps: list[float],
    label_to_id: dict[str, int],
    blank_id: int,
    *,
    frame_confidence: list[float] | np.ndarray | None = None,
    frame_energy: list[float] | np.ndarray | None = None,
) -> ForcedAlignmentResult:
    if logits is None:
        raise ValueError("Forced alignment requires frame-level logits.")

    token_symbols = [token.symbol for token in candidate.ipa.tokens]
    labels = ipa_tokens_to_labels(candidate.ipa.tokens)
    if not labels:
        raise ValueError("Forced alignment requires at least one target label.")

    missing_labels = sorted({label for label in labels if label not in label_to_id})
    if missing_labels:
        raise ValueError(f"Unsupported alignment label(s): {', '.join(missing_labels)}")

    log_probs = np.asarray(logits, dtype=np.float32)
    if log_probs.ndim != 2 or log_probs.shape[0] == 0:
        raise ValueError("Forced alignment requires non-empty 2D frame-level logits.")
    if len(labels) > log_probs.shape[0]:
        raise ValueError(
            f"Too many target labels for forced alignment frames: labels={len(labels)}, frames={log_probs.shape[0]}"
        )

    target_ids = [label_to_id[label] for label in labels]
    vocab_size = log_probs.shape[1]
    invalid_ids = [label for label, label_id in zip(labels, target_ids) if label_id < 0 or label_id >= vocab_size]
    if blank_id < 0 or blank_id >= vocab_size or invalid_ids:
        invalid_text = ", ".join(sorted(set(invalid_ids)))
        raise ValueError(f"Unsupported alignment label(s): {invalid_text or '<blank>'}")
    original_frame_count = log_probs.shape[0]
    log_probs = log_probs - np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
    compressed_log_probs, frame_map, pause_debug = _compress_pause_frames(
        log_probs,
        blank_id,
        min_remaining_frames=len(labels),
        frame_confidence=frame_confidence,
        frame_energy=frame_energy,
    )

    states, best_score = _viterbi_ctc_path(compressed_log_probs, target_ids, blank_id)
    state_log_posteriors, forward_backward_total = _forward_backward_state_posteriors(
        compressed_log_probs,
        target_ids,
        blank_id,
    )
    compressed_frame_count = len(states)
    frame_boundaries = _frame_boundaries(frame_timestamps, original_frame_count)

    label_frame_buckets: list[list[int]] = [[] for _ in labels]
    compressed_label_frame_buckets: list[list[int]] = [[] for _ in labels]
    blank_frames = 0
    extended = _build_extended_sequence(target_ids, blank_id)
    for frame_index, state in enumerate(states):
        symbol_id = extended[state]
        if symbol_id == blank_id:
            blank_frames += 1
            continue
        label_index = (state - 1) // 2
        original_frame_index = int(frame_map[frame_index])
        label_frame_buckets[label_index].append(original_frame_index)
        compressed_label_frame_buckets[label_index].append(frame_index)

    segments: list[AlignmentSegment] = []
    confidences: list[float] = []
    confidence_debug: list[dict] = []
    segment_boundaries = _segment_boundaries(label_frame_buckets, original_frame_count)
    for index, frame_indices in enumerate(label_frame_buckets):
        if not frame_indices:
            continue

        boundary_range = segment_boundaries[index]
        if boundary_range is None:
            continue
        frame_start, frame_end_exclusive = boundary_range
        time_start = frame_boundaries[frame_start]
        time_end = frame_boundaries[min(frame_end_exclusive, len(frame_boundaries) - 1)]

        label_id = target_ids[index]
        confidence, token_confidence_debug = _label_confidence(
            label_index=index,
            label_id=label_id,
            token_symbol=token_symbols[index],
            compressed_frame_indices=compressed_label_frame_buckets[index],
            log_probs=compressed_log_probs,
            state_log_posteriors=state_log_posteriors,
        )
        token_confidence_debug.update(
            {
                "index": index,
                "token": token_symbols[index],
                "label": labels[index],
            }
        )
        confidence_debug.append(token_confidence_debug)
        confidences.append(confidence)
        segments.append(
            AlignmentSegment(
                token=token_symbols[index],
                label=labels[index],
                start_time=time_start,
                end_time=time_end,
                frame_start=frame_start,
                frame_end=max(frame_start, frame_end_exclusive - 1),
                confidence=confidence,
            )
        )

    coverage = len(segments) / len(labels) if labels else 0.0
    avg_token_confidence = float(np.mean(confidences)) if confidences else 0.0
    blank_ratio = blank_frames / compressed_frame_count if compressed_frame_count else 1.0
    normalized_log_prob = best_score / max(compressed_frame_count, 1)
    alignment_debug = {
        "pause_compression": pause_debug,
        "confidence_method": "viterbi_path_with_forward_backward_posterior",
        "forward_backward_total_log_prob": forward_backward_total,
        "confidence_details": confidence_debug[:20],
        "original_frame_count": original_frame_count,
        "alignment_frame_count": compressed_frame_count,
    }

    return ForcedAlignmentResult(
        labels=labels,
        token_symbols=token_symbols,
        segments=segments,
        total_log_prob=best_score,
        normalized_log_prob=normalized_log_prob,
        avg_token_confidence=avg_token_confidence,
        coverage=coverage,
        blank_ratio=blank_ratio,
        alignment_debug=alignment_debug,
    )
