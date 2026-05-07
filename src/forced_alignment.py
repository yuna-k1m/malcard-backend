from __future__ import annotations

"""Frame-level CTC forced alignment for reference IPA tokens."""

from math import inf

import numpy as np

from src.label_to_ipa import ipa_tokens_to_labels
from src.types import AlignmentSegment, ForcedAlignmentResult, PronunciationCandidate


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

    dp = np.full((num_frames, num_states), -inf, dtype=np.float64)
    back = np.full((num_frames, num_states), -1, dtype=np.int32)

    dp[0, 0] = float(log_probs[0, blank_id])
    if num_states > 1:
        dp[0, 1] = float(log_probs[0, extended[1]])

    for frame in range(1, num_frames):
        for state in range(num_states):
            candidates = [(dp[frame - 1, state], state)]
            if state - 1 >= 0:
                candidates.append((dp[frame - 1, state - 1], state - 1))
            if (
                state - 2 >= 0
                and extended[state] != blank_id
                and extended[state] != extended[state - 2]
            ):
                candidates.append((dp[frame - 1, state - 2], state - 2))

            best_score, best_prev = max(candidates, key=lambda item: item[0])
            dp[frame, state] = best_score + float(log_probs[frame, extended[state]])
            back[frame, state] = best_prev

    end_candidates = [(dp[num_frames - 1, num_states - 1], num_states - 1)]
    if num_states > 1:
        end_candidates.append((dp[num_frames - 1, num_states - 2], num_states - 2))
    best_score, best_state = max(end_candidates, key=lambda item: item[0])

    states = [best_state]
    state = best_state
    for frame in range(num_frames - 1, 0, -1):
        state = int(back[frame, state])
        states.append(state)
    states.reverse()
    return states, float(best_score)


def force_align_candidate(
    candidate: PronunciationCandidate,
    logits: list[list[float]],
    frame_timestamps: list[float],
    label_to_id: dict[str, int],
    blank_id: int,
) -> ForcedAlignmentResult:
    if not logits:
        raise ValueError("Forced alignment requires frame-level logits.")

    token_symbols = [token.symbol for token in candidate.ipa.tokens]
    labels = ipa_tokens_to_labels(candidate.ipa.tokens)
    target_ids = [label_to_id[label] for label in labels]

    log_probs = np.asarray(logits, dtype=np.float64)
    log_probs = log_probs - np.logaddexp.reduce(log_probs, axis=1, keepdims=True)

    states, best_score = _viterbi_ctc_path(log_probs, target_ids, blank_id)
    num_frames = len(states)

    label_frame_buckets: list[list[int]] = [[] for _ in labels]
    blank_frames = 0
    extended = _build_extended_sequence(target_ids, blank_id)
    for frame_index, state in enumerate(states):
        symbol_id = extended[state]
        if symbol_id == blank_id:
            blank_frames += 1
            continue
        label_index = (state - 1) // 2
        label_frame_buckets[label_index].append(frame_index)

    segments: list[AlignmentSegment] = []
    confidences: list[float] = []
    for index, frame_indices in enumerate(label_frame_buckets):
        if not frame_indices:
            continue

        frame_start = frame_indices[0]
        frame_end = frame_indices[-1]
        time_start = frame_timestamps[frame_start] if frame_timestamps else float(frame_start)
        if frame_timestamps and frame_end + 1 < len(frame_timestamps):
            time_end = frame_timestamps[frame_end + 1]
        elif frame_timestamps:
            frame_width = frame_timestamps[1] - frame_timestamps[0] if len(frame_timestamps) > 1 else 0.02
            time_end = frame_timestamps[frame_end] + frame_width
        else:
            time_end = float(frame_end + 1)

        label_id = target_ids[index]
        probs = np.exp(log_probs[frame_indices, label_id])
        confidence = float(np.mean(probs))
        confidences.append(confidence)
        segments.append(
            AlignmentSegment(
                token=token_symbols[index],
                label=labels[index],
                start_time=time_start,
                end_time=time_end,
                frame_start=frame_start,
                frame_end=frame_end,
                confidence=confidence,
            )
        )

    coverage = len(segments) / len(labels) if labels else 0.0
    avg_token_confidence = float(np.mean(confidences)) if confidences else 0.0
    blank_ratio = blank_frames / num_frames if num_frames else 1.0
    normalized_log_prob = best_score / max(num_frames, 1)

    return ForcedAlignmentResult(
        labels=labels,
        token_symbols=token_symbols,
        segments=segments,
        total_log_prob=best_score,
        normalized_log_prob=normalized_log_prob,
        avg_token_confidence=avg_token_confidence,
        coverage=coverage,
        blank_ratio=blank_ratio,
    )
