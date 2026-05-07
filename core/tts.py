"""Google Cloud Text-to-Speech 래퍼.

인증 설정 (최초 1회):
    gcloud auth application-default login

또는 서비스 계정 사용 시:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from google.cloud import texttospeech

_CLIENT: texttospeech.TextToSpeechClient | None = None


def _get_client() -> texttospeech.TextToSpeechClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = texttospeech.TextToSpeechClient()
    return _CLIENT


def generate_tts(
    text: str,
    cache_dir: str | Path,
    voice: str = "ko-KR-Neural2-A",  # Neural2 > Wavenet > Standard 순으로 품질 높음
    speed: float = 1.0,              # 0.25 ~ 4.0
) -> Path:
    """텍스트 → WAV 변환. 동일 (text, voice) 조합은 캐시 파일 반환.

    Args:
        text: 합성할 한국어 텍스트
        cache_dir: WAV 캐시 저장 디렉토리
        voice: Google TTS 음성 이름 (ko-KR-Neural2-A/B/C/D)
        speed: 발화 속도 배율

    Returns:
        생성된 WAV 파일 경로 (16kHz mono PCM)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha256(f"{text}:{voice}:{speed}".encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{key}.wav"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    response = _get_client().synthesize_speech(
        input=texttospeech.SynthesisInput(text=text),
        voice=texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name=voice,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            speaking_rate=speed,
        ),
    )
    cache_path.write_bytes(response.audio_content)
    return cache_path