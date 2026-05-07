from __future__ import annotations

"""Richer audio-recognition wrapper for alignment-driven evaluation."""

from pathlib import Path

import torch

from src.audio_to_ipa import AudioToIPARecognizer
from src.label_to_ipa import decode_label_text
from src.quality import analyze_audio_quality
from src.types import AudioRecognitionResult


def recognize_audio(recognizer: AudioToIPARecognizer, audio_path: str | Path) -> AudioRecognitionResult:
    """Run the recognizer and keep metadata needed for later constrained decoding."""

    loaded = recognizer.load_audio(audio_path, target_sr=16000)
    if isinstance(loaded, tuple):
        audio, sampling_rate = loaded
    else:
        audio, sampling_rate = loaded, 16000
    quality_report = analyze_audio_quality(audio, sampling_rate)
    inputs = recognizer.processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(recognizer.device)
    attention_mask = inputs.attention_mask.to(recognizer.device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits_tensor = recognizer.model(input_values=input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits_tensor, dim=-1)
    raw_label_text = recognizer.processor.batch_decode(pred_ids)[0]
    raw_label_text = raw_label_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
    raw_labels, sequence = decode_label_text(raw_label_text)

    logits = logits_tensor[0].detach().cpu()
    frame_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
    duration = len(audio) / float(sampling_rate)
    frame_timestamps = [duration * index / max(1, len(frame_confidence) - 1) for index in range(len(frame_confidence))]

    return AudioRecognitionResult(
        raw_text=sequence.raw_text,
        normalized_text=sequence.normalized_text,
        tokens=sequence.tokens,
        raw_label_text=raw_label_text,
        raw_labels=raw_labels,
        logits=logits.tolist(),
        frame_confidence=frame_confidence,
        frame_timestamps=frame_timestamps,
        sampling_rate=sampling_rate,
        quality_report=quality_report,
    )
