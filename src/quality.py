from __future__ import annotations

"""Audio quality and forced-alignment confidence gates."""

import math

import numpy as np

from src.types import AlignmentConfidenceReport, AudioQualityReport, ForcedAlignmentResult


def analyze_audio_quality(audio: np.ndarray, sampling_rate: int) -> AudioQualityReport:
    duration_sec = len(audio) / float(sampling_rate) if sampling_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if audio.size else 1.0

    frame_size = max(1, int(sampling_rate * 0.02))
    frames = [audio[index:index + frame_size] for index in range(0, len(audio), frame_size)] or [audio]
    frame_energy = [float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0 for frame in frames]
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


def assess_alignment_confidence(result: ForcedAlignmentResult) -> AlignmentConfidenceReport:
    reasons: list[str] = []
    if result.coverage < 0.85:
        reasons.append("정답 음소 대부분이 시간축에 안정적으로 배치되지 않았습니다.")
    if result.avg_token_confidence < 0.20:
        reasons.append("정렬 경로의 음소 신뢰도가 낮습니다.")
    if result.normalized_log_prob < -4.5:
        reasons.append("전체 정렬 경로의 로그확률이 낮습니다.")

    message = "forced alignment를 신뢰할 수 있습니다." if not reasons else " ".join(reasons)
    return AlignmentConfidenceReport(
        passed=not reasons,
        avg_token_confidence=result.avg_token_confidence,
        coverage=result.coverage,
        normalized_log_prob=result.normalized_log_prob,
        message=message,
    )
