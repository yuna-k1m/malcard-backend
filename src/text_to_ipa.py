from __future__ import annotations

"""Utilities for converting pronunciation-form Korean text into IPA."""

import os
import re
from pathlib import Path

try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
except Exception:  # pragma: no cover
    EspeakWrapper = None

try:
    from IPAkor.transcription import UniTranscript
except Exception:  # pragma: no cover
    UniTranscript = None

from src.ipa_utils import build_ipa_sequence, normalize_ipa_text
from src.types import IPASequence


_transcriber = None

_ONSET_IPA = {
    "ㄱ": "k", "ㄲ": "k͈", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t͈", "ㄹ": "ɾ",
    "ㅁ": "m", "ㅂ": "p", "ㅃ": "p͈", "ㅅ": "s", "ㅆ": "s͈", "ㅇ": "",
    "ㅈ": "tɕ", "ㅉ": "tɕ͈", "ㅊ": "tɕʰ", "ㅋ": "kʰ", "ㅌ": "tʰ", "ㅍ": "pʰ", "ㅎ": "h",
}
_VOWEL_IPA = {
    "ㅏ": "a", "ㅐ": "ɛ", "ㅑ": "ja", "ㅒ": "jɛ", "ㅓ": "ʌ", "ㅔ": "e",
    "ㅕ": "jʌ", "ㅖ": "je", "ㅗ": "o", "ㅘ": "wa", "ㅙ": "wɛ", "ㅚ": "ø",
    "ㅛ": "jo", "ㅜ": "u", "ㅝ": "wʌ", "ㅞ": "we", "ㅟ": "y", "ㅠ": "ju",
    "ㅡ": "ɯ", "ㅢ": "ɯi", "ㅣ": "i",
}
_CODA_IPA = {
    "": "", "ㄱ": "k̚", "ㄲ": "k̚", "ㄳ": "k̚", "ㄴ": "n", "ㄵ": "n", "ㄶ": "n",
    "ㄷ": "t̚", "ㄹ": "l", "ㄺ": "lk̚", "ㄻ": "lm", "ㄼ": "lp̚", "ㄽ": "ls",
    "ㄾ": "ltʰ", "ㄿ": "lpʰ", "ㅀ": "lh", "ㅁ": "m", "ㅂ": "p̚", "ㅄ": "p̚",
    "ㅅ": "t̚", "ㅆ": "t̚", "ㅇ": "ŋ", "ㅈ": "t̚", "ㅊ": "t̚", "ㅋ": "k̚",
    "ㅌ": "t̚", "ㅍ": "p̚", "ㅎ": "h",
}


def _configure_espeak() -> None:
    if EspeakWrapper is None:
        return

    dll_candidates = [
        os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"),
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
        r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
    ]
    for dll_path in dll_candidates:
        if dll_path and Path(dll_path).exists():
            EspeakWrapper.set_library(dll_path)
            return


def get_transcriber():
    global _transcriber
    if _transcriber is None and UniTranscript is not None:
        try:
            _configure_espeak()
            _transcriber = UniTranscript()
        except Exception:
            _transcriber = None
    return _transcriber


def normalize_korean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _decompose_syllable(char: str) -> tuple[str, str, str] | None:
    if not ("가" <= char <= "힣"):
        return None
    code = ord(char) - ord("가")
    onset = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"][code // 588]
    vowel = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"][(code % 588) // 28]
    coda = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"][code % 28]
    return onset, vowel, coda


def _manual_ipa_from_pronunciation(pronunciation: str) -> IPASequence:
    positions: list[str] = []
    raw_parts: list[str] = []
    for char in pronunciation:
        if char.isspace():
            raw_parts.append(" ")
            continue
        decomposed = _decompose_syllable(char)
        if decomposed is None:
            raw_parts.append(char)
            positions.append("other")
            continue
        onset, vowel, coda = decomposed
        onset_ipa = _ONSET_IPA[onset]
        if onset_ipa:
            raw_parts.append(onset_ipa)
            positions.append("onset")
        raw_parts.append(_VOWEL_IPA[vowel])
        positions.append("nucleus")
        coda_ipa = _CODA_IPA[coda]
        if coda_ipa:
            raw_parts.append(coda_ipa)
            positions.append("coda")

    raw_text = " ".join(part for part in raw_parts if part != " ").replace("  ", " ").strip()
    sequence = build_ipa_sequence(raw_text)
    if len(sequence.tokens) == len(positions):
        for token, syllable_position in zip(sequence.tokens, positions):
            token.syllable_position = syllable_position
    return sequence


def pronunciation_to_ipa(pronunciation: str) -> IPASequence:
    pronunciation = normalize_korean_text(pronunciation)
    if not pronunciation:
        return build_ipa_sequence("")

    manual_sequence = _manual_ipa_from_pronunciation(pronunciation)
    transcriber = get_transcriber()
    if transcriber is None:
        return manual_sequence

    try:
        raw_ipa = transcriber.transcribator(pronunciation).strip()
    except Exception:
        return manual_sequence
    if not raw_ipa:
        return manual_sequence

    return IPASequence(
        raw_text=raw_ipa,
        normalized_text=normalize_ipa_text(raw_ipa),
        tokens=manual_sequence.tokens,
    )


def text_to_ipa(text: str) -> str:
    return pronunciation_to_ipa(text).normalized_text
