from __future__ import annotations

import re
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


MODEL_ID = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"


class AudioToIPARecognizer:
    def __init__(self, model_id: str = MODEL_ID):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def load_audio(self, audio_path: str | Path, target_sr: int = 16000) -> np.ndarray:
        audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        if audio.size == 0:
            raise ValueError("빈 오디오입니다.")
        audio = audio.astype(np.float32)
        max_abs = np.max(np.abs(audio))
        if max_abs > 0:
            audio = audio / max_abs
        return audio

    def audio_to_ipa(self, audio_path: str | Path) -> str:
        audio = self.load_audio(audio_path, target_sr=16000)

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = self.model(
                input_values=input_values,
                attention_mask=attention_mask
            ).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred = self.processor.batch_decode(pred_ids)[0]

        pred = pred.strip()
        pred = re.sub(r"\s+", " ", pred)
        pred = pred.replace("<pad>", "").replace("</s>", "").replace("<s>", "")
        pred = pred.strip()

        return pred