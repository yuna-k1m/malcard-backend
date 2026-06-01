"""gTTS (Google Translate TTS) 한국어 음성 합성 — 무료, 인터넷 필요.

gTTS는 mp3를 출력하므로 librosa로 16kHz mono PCM wav로 변환한다.
(MMS-TTS는 torch 2.11 ↔ transformers 4.44 weight_norm 키 mismatch로 출력 불가라
 gTTS로 대체. torch와 무관해 버전 충돌 없음.)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import scipy.io.wavfile

SAMPLE_RATE = 16000


def generate_tts(
    text: str,
    cache_dir: str | Path,
    speed: float = 1.0,
    filename: str | None = None,
) -> Path:
    """텍스트 → WAV(16kHz mono PCM). 동일 (text, speed)는 캐시 파일 반환.

    Args:
        text: 합성할 한국어 텍스트
        cache_dir: WAV 캐시 저장 디렉토리
        speed: <1.0이면 gTTS slow 모드 (gTTS는 slow on/off만 지원)
        filename: 명시하면 그 이름으로 저장. None이면 해시 키.

    Returns:
        WAV 파일 경로 (16kHz mono PCM)
    """
    from gtts import gTTS
    import librosa

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        cache_path = cache_dir / filename
    else:
        key = hashlib.sha256(f"{text}:{speed}".encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{key}.wav"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    mp3_path = cache_path.with_suffix(".mp3")
    gTTS(text=text, lang="ko", slow=(speed < 1.0)).save(str(mp3_path))

    audio, _ = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    scipy.io.wavfile.write(str(cache_path), SAMPLE_RATE, audio_int16)
    mp3_path.unlink(missing_ok=True)
    return cache_path
