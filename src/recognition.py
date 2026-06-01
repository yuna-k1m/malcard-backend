from __future__ import annotations

"""Richer audio-recognition wrapper for alignment-driven evaluation."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from src.audio_to_ipa import AudioToIPARecognizer
from src.label_to_ipa import decode_label_text
from src.quality import analyze_audio_quality, trim_audio_edges
from src.types import AudioRecognitionResult


def _logit_frame_energy(audio: np.ndarray, frame_count: int) -> list[float]:
    if frame_count <= 0:
        return []
    energies: list[float] = []
    sample_count = len(audio)
    for index in range(frame_count):
        start = int(round(sample_count * index / frame_count))
        end = int(round(sample_count * (index + 1) / frame_count))
        frame = audio[start:max(start + 1, end)]
        energies.append(float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0)
    return energies


def recognize_audio(recognizer: AudioToIPARecognizer, audio_path: str | Path) -> AudioRecognitionResult:
    """Run the recognizer and keep metadata needed for later constrained decoding."""

    loaded = recognizer.load_audio(audio_path, target_sr=16000)
    if isinstance(loaded, tuple):
        audio, sampling_rate = loaded
    else:
        audio, sampling_rate = loaded, 16000
    alignment_audio, trim_start_sample, trim_end_sample = trim_audio_edges(audio, sampling_rate)
    quality_report = analyze_audio_quality(alignment_audio, sampling_rate)
    inputs = recognizer.processor(alignment_audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(recognizer.device)
    attention_mask = inputs.attention_mask.to(recognizer.device) if "attention_mask" in inputs else None

    inference_context = getattr(recognizer, "inference_context", nullcontext)
    with torch.inference_mode(), inference_context():
        logits_tensor = recognizer.model(input_values=input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits_tensor, dim=-1)
    raw_label_text = recognizer.processor.batch_decode(pred_ids)[0]
    raw_label_text = raw_label_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
    raw_labels, sequence = decode_label_text(raw_label_text)

    logits = logits_tensor[0].detach().to(dtype=torch.float32).cpu()
    frame_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
    duration = len(alignment_audio) / float(sampling_rate)
    # Store frame boundaries, not centers. There are N logit frames and N+1
    # boundaries, so the last segment can end exactly at the audio duration.
    frame_count = len(frame_confidence)
    frame_timestamps = [
        duration * index / max(1, frame_count)
        for index in range(frame_count + 1)
    ]
    frame_energy = _logit_frame_energy(alignment_audio, frame_count)

    return AudioRecognitionResult(
        raw_text=sequence.raw_text,
        normalized_text=sequence.normalized_text,
        tokens=sequence.tokens,
        raw_label_text=raw_label_text,
        raw_labels=raw_labels,
        logits=logits.numpy(),
        frame_confidence=frame_confidence,
        frame_energy=frame_energy,
        frame_timestamps=frame_timestamps,
        sampling_rate=sampling_rate,
        quality_report=quality_report,
        trim_start_sec=trim_start_sample / float(sampling_rate),
        trim_end_sec=trim_end_sample / float(sampling_rate),
        original_duration_sec=len(audio) / float(sampling_rate),
        trimmed_duration_sec=duration,
    )
