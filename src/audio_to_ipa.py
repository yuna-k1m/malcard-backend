from __future__ import annotations

import re
from contextlib import nullcontext
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC


MODEL_ID = "slplab/wav2vec2-xls-r-300m_phone-mfa_korean"


class AudioToIPARecognizer:
    def __init__(self, model_id: str = MODEL_ID, *, use_cuda_autocast: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda_autocast = bool(use_cuda_autocast and self.device == "cuda")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self._alignment_vocab: dict[str, int] | None = None
        self._alignment_blank_id: int | None = None

    def inference_context(self):
        if self.use_cuda_autocast:
            return torch.autocast(device_type="cuda")
        return nullcontext()

    def get_alignment_vocab(self) -> tuple[dict[str, int], int]:
        if self._alignment_vocab is None or self._alignment_blank_id is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Recognizer processor does not expose a tokenizer for forced alignment.")
            self._alignment_vocab = tokenizer.get_vocab()
            self._alignment_blank_id = tokenizer.pad_token_id
        return self._alignment_vocab, self._alignment_blank_id

    def load_audio(self, audio_path: str | Path, target_sr: int = 16000) -> np.ndarray:
        try:
            audio, sr = sf.read(str(audio_path), always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        except Exception:
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

        with torch.inference_mode(), self.inference_context():
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
