from __future__ import annotations

"""Korean pronunciation-form Hangul to IPA conversion.

This module intentionally avoids lossy symbol collapsing. It converts a
pronunciation-form Korean string into a detailed, consistent IPA space that is
compatible with the user-side label-to-IPA conversion.
"""

import re

from src.ipa_utils import build_ipa_sequence
from src.types import IPASequence


ONSETS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VOWELS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
CODAS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

ONSET_IPA = {
    "ㄱ": "k", "ㄲ": "k͈", "ㄴ": "n", "ㄷ": "t", "ㄸ": "t͈", "ㄹ": "ɾ",
    "ㅁ": "m", "ㅂ": "p", "ㅃ": "p͈", "ㅅ": "s", "ㅆ": "s͈", "ㅇ": "",
    "ㅈ": "tɕ", "ㅉ": "tɕ͈", "ㅊ": "tɕʰ", "ㅋ": "kʰ", "ㅌ": "tʰ", "ㅍ": "pʰ", "ㅎ": "h",
}
CODA_IPA = {
    "": "", "ㄱ": "k̚", "ㄲ": "k̚", "ㄳ": "k̚", "ㄴ": "n", "ㄵ": "n", "ㄶ": "n",
    "ㄷ": "t̚", "ㄹ": "l", "ㄺ": "k̚", "ㄻ": "m", "ㄼ": "p̚", "ㄽ": "l", "ㄾ": "l", "ㄿ": "p̚", "ㅀ": "l",
    "ㅁ": "m", "ㅂ": "p̚", "ㅄ": "p̚", "ㅅ": "t̚", "ㅆ": "t̚", "ㅇ": "ŋ", "ㅈ": "t̚", "ㅊ": "t̚", "ㅋ": "k̚",
    "ㅌ": "t̚", "ㅍ": "p̚", "ㅎ": "t̚",
}
VOWEL_IPA = {
    "ㅏ": "a", "ㅐ": "ɛ", "ㅑ": "ja", "ㅒ": "jɛ", "ㅓ": "ʌ", "ㅔ": "e",
    "ㅕ": "jʌ", "ㅖ": "je", "ㅗ": "o", "ㅘ": "wa", "ㅙ": "wɛ", "ㅚ": "we",
    "ㅛ": "jo", "ㅜ": "u", "ㅝ": "wʌ", "ㅞ": "we", "ㅟ": "wi", "ㅠ": "ju",
    "ㅡ": "ɯ", "ㅢ": "ɰi", "ㅣ": "i",
}

SINO_DIGITS = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SINO_SMALL_UNITS = ["", "십", "백", "천"]
SINO_BIG_UNITS = ["", "만", "억", "조"]
NATIVE_HOUR_MAP = {
    1: "한",
    2: "두",
    3: "세",
    4: "네",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
    10: "열",
    11: "열한",
    12: "열두",
}


def _four_digits_to_sino_korean(chunk: str) -> str:
    chunk = chunk.zfill(4)
    parts: list[str] = []
    for index, digit_char in enumerate(chunk):
        digit = int(digit_char)
        if digit == 0:
            continue
        unit_index = 3 - index
        if digit == 1 and unit_index > 0:
            digit_text = ""
        else:
            digit_text = SINO_DIGITS[digit]
        parts.append(digit_text + SINO_SMALL_UNITS[unit_index])
    return "".join(parts)


def number_to_sino_korean(number_text: str) -> str:
    stripped = number_text.lstrip("0")
    if not stripped:
        return SINO_DIGITS[0]
    if number_text.startswith("0") and len(number_text) > 1:
        return "".join(SINO_DIGITS[int(ch)] for ch in number_text)

    groups: list[str] = []
    while stripped:
        groups.append(stripped[-4:])
        stripped = stripped[:-4]

    parts: list[str] = []
    for index, group in enumerate(groups):
        converted = _four_digits_to_sino_korean(group)
        if not converted:
            continue
        parts.append(converted + SINO_BIG_UNITS[index])
    return "".join(reversed(parts))


def _replace_number_with_korean(match: re.Match[str]) -> str:
    number_text = match.group(0)
    following = match.string[match.end():]
    number = int(number_text)

    # Clock-hour style usage keeps native Korean numerals for 1~12.
    if following.startswith("시") and 1 <= number <= 12:
        return NATIVE_HOUR_MAP[number]

    return number_to_sino_korean(number_text)


def normalize_korean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\d+", _replace_number_with_korean, text)
    text = re.sub(r"\s+", " ", text)
    return text


def decompose_syllable(char: str) -> tuple[str, str, str] | None:
    if not ("가" <= char <= "힣"):
        return None
    code = ord(char) - ord("가")
    onset = ONSETS[code // 588]
    vowel = VOWELS[(code % 588) // 28]
    coda = CODAS[code % 28]
    return onset, vowel, coda


def _s_before_high_front(vowel: str) -> bool:
    return vowel in {"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ"}


def _onset_to_ipa(onset: str, vowel: str) -> str:
    ipa = ONSET_IPA[onset]
    if onset == "ㅅ" and _s_before_high_front(vowel):
        return "ɕ"
    if onset == "ㅆ" and _s_before_high_front(vowel):
        return "ɕ͈"
    return ipa


def pronunciation_to_ipa(pronunciation: str) -> IPASequence:
    pronunciation = normalize_korean_text(pronunciation)
    if not pronunciation:
        return build_ipa_sequence("")

    symbols: list[tuple[str, str]] = []

    for char in pronunciation:
        if char.isspace():
            continue

        decomposed = decompose_syllable(char)
        if decomposed is None:
            # punctuation, Latin letters, etc.
            symbols.append((char, "other"))
            continue

        onset, vowel, coda = decomposed
        onset_ipa = _onset_to_ipa(onset, vowel)
        vowel_ipa = VOWEL_IPA[vowel]
        coda_ipa = CODA_IPA[coda]

        if onset_ipa:
            symbols.append((onset_ipa, "onset"))

        symbols.append((vowel_ipa, "nucleus"))

        if coda_ipa:
            symbols.append((coda_ipa, "coda"))

    raw_ipa = " ".join(symbol for symbol, _ in symbols)
    sequence = build_ipa_sequence(raw_ipa)

    # Important:
    # scoring.py relies on syllable_position == "coda"
    # to calculate coda-specific scores.
    if len(sequence.tokens) == len(symbols):
        for token, (_, position) in zip(sequence.tokens, symbols):
            token.syllable_position = position
    else:
        # Fallback: even if tokenization length changes unexpectedly,
        # mark unreleased final stops and final l as coda-like.
        for token in sequence.tokens:
            if token.symbol.endswith("̚") or token.symbol == "l":
                token.syllable_position = "coda"
            elif token.category == "vowel":
                token.syllable_position = "nucleus"
            elif token.syllable_position == "unknown":
                token.syllable_position = "onset"

    return sequence


def text_to_ipa(text: str) -> str:
    return pronunciation_to_ipa(text).normalized_text
