from __future__ import annotations

"""IPA normalization and tokenization helpers."""

import re
import unicodedata

import regex

from src.types import IPASequence, PhoneToken


STRESS_MARKS = {"ˈ", "ˌ"}
TOKEN_MODIFIERS = {
    "ʰ", "ʷ", "ʲ", "ː", "ˑ", "̚", "͈", "̥", "̬", "̹", "̜", "̟", "̠",
    "̈", "̩", "̃", "̯", "̤", "̰", "̪", "̺", "̻",
}
VOWEL_BASES = set("aeiouyɯɨʉɪʊøœɛɜɞəɐɔæɑɒʌ")
SEMIVOWELS = {"j", "w", "ɰ", "ɥ"}


def normalize_ipa_text(ipa: str) -> str:
    if not ipa:
        return ""

    text = unicodedata.normalize("NFC", ipa)
    for mark in STRESS_MARKS:
        text = text.replace(mark, "")

    text = re.sub(r"[,\.\?!;:\"'`\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _categorize_symbol(symbol: str) -> str:
    if not symbol:
        return "other"
    if any(ch in VOWEL_BASES for ch in symbol):
        return "vowel"
    if symbol in SEMIVOWELS:
        return "semivowel"
    if symbol in {" ", "|"}:
        return "boundary"
    return "consonant"


def ipa_to_tokens(ipa: str) -> list[PhoneToken]:
    ipa = normalize_ipa_text(ipa)
    if not ipa:
        return []

    if " " in ipa:
        return [PhoneToken(symbol=chunk, category=_categorize_symbol(chunk)) for chunk in ipa.split() if chunk]

    tokens: list[str] = []
    clusters = regex.findall(r"\X", ipa)
    for cluster in clusters:
        if cluster in TOKEN_MODIFIERS and tokens:
            tokens[-1] = tokens[-1] + cluster
        else:
            tokens.append(cluster)

    return [PhoneToken(symbol=token, category=_categorize_symbol(token)) for token in tokens]


def build_ipa_sequence(ipa: str) -> IPASequence:
    normalized = normalize_ipa_text(ipa)
    return IPASequence(raw_text=ipa.strip(), normalized_text=normalized, tokens=ipa_to_tokens(normalized))


def pretty_tokens(tokens: list[PhoneToken]) -> str:
    return " ".join(token.symbol for token in tokens)
