from __future__ import annotations

"""Explainable phone cost model for Korean IPA alignment."""

from dataclasses import dataclass

from src.types import PhoneToken


@dataclass(frozen=True)
class PhoneFeatures:
    phone: str
    category: str
    place: str = "other"
    manner: str = "other"
    voicing: str = "other"
    aspiration: str = "none"
    tenseness: str = "none"
    height: str = "other"
    backness: str = "other"
    rounding: str = "other"


PHONE_FEATURES: dict[str, PhoneFeatures] = {
    "k": PhoneFeatures("k", "consonant", "velar", "stop", "voiceless"),
    "k̚": PhoneFeatures("k̚", "consonant", "velar", "stop", "voiceless"),
    "kʰ": PhoneFeatures("kʰ", "consonant", "velar", "stop", "voiceless", aspiration="strong"),
    "k͈": PhoneFeatures("k͈", "consonant", "velar", "stop", "voiceless", tenseness="tense"),
    "t": PhoneFeatures("t", "consonant", "alveolar", "stop", "voiceless"),
    "t̚": PhoneFeatures("t̚", "consonant", "alveolar", "stop", "voiceless"),
    "tʰ": PhoneFeatures("tʰ", "consonant", "alveolar", "stop", "voiceless", aspiration="strong"),
    "t͈": PhoneFeatures("t͈", "consonant", "alveolar", "stop", "voiceless", tenseness="tense"),
    "p": PhoneFeatures("p", "consonant", "bilabial", "stop", "voiceless"),
    "p̚": PhoneFeatures("p̚", "consonant", "bilabial", "stop", "voiceless"),
    "pʰ": PhoneFeatures("pʰ", "consonant", "bilabial", "stop", "voiceless", aspiration="strong"),
    "p͈": PhoneFeatures("p͈", "consonant", "bilabial", "stop", "voiceless", tenseness="tense"),
    "s": PhoneFeatures("s", "consonant", "alveolar", "fricative", "voiceless"),
    "s͈": PhoneFeatures("s͈", "consonant", "alveolar", "fricative", "voiceless", tenseness="tense"),
    "ɕ": PhoneFeatures("ɕ", "consonant", "alveolo-palatal", "fricative", "voiceless"),
    "ɕ͈": PhoneFeatures("ɕ͈", "consonant", "alveolo-palatal", "fricative", "voiceless", tenseness="tense"),
    "h": PhoneFeatures("h", "consonant", "glottal", "fricative", "voiceless"),
    "tɕ": PhoneFeatures("tɕ", "consonant", "alveolo-palatal", "affricate", "voiceless"),
    "tɕʰ": PhoneFeatures("tɕʰ", "consonant", "alveolo-palatal", "affricate", "voiceless", aspiration="strong"),
    "tɕ͈": PhoneFeatures("tɕ͈", "consonant", "alveolo-palatal", "affricate", "voiceless", tenseness="tense"),
    "n": PhoneFeatures("n", "consonant", "alveolar", "nasal", "voiced"),
    "m": PhoneFeatures("m", "consonant", "bilabial", "nasal", "voiced"),
    "ŋ": PhoneFeatures("ŋ", "consonant", "velar", "nasal", "voiced"),
    "ɾ": PhoneFeatures("ɾ", "consonant", "alveolar", "tap", "voiced"),
    "l": PhoneFeatures("l", "consonant", "alveolar", "lateral", "voiced"),
    "r": PhoneFeatures("r", "consonant", "alveolar", "trill", "voiced"),
    "j": PhoneFeatures("j", "semivowel", "palatal", "approximant", "voiced"),
    "w": PhoneFeatures("w", "semivowel", "labio-velar", "approximant", "voiced"),
    "ɰ": PhoneFeatures("ɰ", "semivowel", "velar", "approximant", "voiced"),
    "a": PhoneFeatures("a", "vowel", height="open", backness="front", rounding="unrounded"),
    "ɛ": PhoneFeatures("ɛ", "vowel", height="open-mid", backness="front", rounding="unrounded"),
    "e": PhoneFeatures("e", "vowel", height="close-mid", backness="front", rounding="unrounded"),
    "i": PhoneFeatures("i", "vowel", height="close", backness="front", rounding="unrounded"),
    "ʌ": PhoneFeatures("ʌ", "vowel", height="open-mid", backness="back", rounding="unrounded"),
    "o": PhoneFeatures("o", "vowel", height="close-mid", backness="back", rounding="rounded"),
    "u": PhoneFeatures("u", "vowel", height="close", backness="back", rounding="rounded"),
    "ɯ": PhoneFeatures("ɯ", "vowel", height="close", backness="back", rounding="unrounded"),
    "ø": PhoneFeatures("ø", "vowel", height="close-mid", backness="front", rounding="rounded"),
    "y": PhoneFeatures("y", "vowel", height="close", backness="front", rounding="rounded"),
    "we": PhoneFeatures("we", "vowel", height="close-mid", backness="back", rounding="rounded"),
    "wa": PhoneFeatures("wa", "vowel", height="open", backness="back", rounding="rounded"),
    "wʌ": PhoneFeatures("wʌ", "vowel", height="open-mid", backness="back", rounding="rounded"),
    "wɛ": PhoneFeatures("wɛ", "vowel", height="open-mid", backness="front", rounding="rounded"),
    "ja": PhoneFeatures("ja", "vowel", height="open", backness="front", rounding="unrounded"),
    "jʌ": PhoneFeatures("jʌ", "vowel", height="open-mid", backness="back", rounding="unrounded"),
    "jo": PhoneFeatures("jo", "vowel", height="close-mid", backness="back", rounding="rounded"),
    "ju": PhoneFeatures("ju", "vowel", height="close", backness="back", rounding="rounded"),
    "je": PhoneFeatures("je", "vowel", height="close-mid", backness="front", rounding="unrounded"),
    "jɛ": PhoneFeatures("jɛ", "vowel", height="open-mid", backness="front", rounding="unrounded"),
    "ɯi": PhoneFeatures("ɯi", "vowel", height="close", backness="back", rounding="unrounded"),
}

PROFILE_PAIR_OVERRIDES = {
    "default": {
        frozenset({"tɕ", "tɕʰ"}): -0.15,
        frozenset({"tɕ", "tɕ͈"}): -0.15,
        frozenset({"tɕʰ", "tɕ͈"}): -0.15,
        frozenset({"k", "kʰ"}): -0.12,
        frozenset({"k", "k͈"}): -0.12,
        frozenset({"t", "tʰ"}): -0.12,
        frozenset({"t", "t͈"}): -0.12,
        frozenset({"p", "pʰ"}): -0.12,
        frozenset({"p", "p͈"}): -0.12,
        frozenset({"s", "s͈"}): -0.10,
        frozenset({"ɾ", "l"}): -0.15,
        frozenset({"ʌ", "o"}): -0.10,
        frozenset({"ɯ", "u"}): -0.10,
        frozenset({"ɛ", "e"}): -0.08,
    },
    "ru": {
        frozenset({"ɾ", "l"}): -0.25,
        frozenset({"tɕ", "tɕʰ"}): -0.08,
        frozenset({"tɕ", "tɕ͈"}): -0.08,
        frozenset({"ʌ", "o"}): -0.10,
        frozenset({"ɯ", "u"}): -0.10,
        frozenset({"s", "s͈"}): -0.08,
    },
}


def get_phone_features(phone: PhoneToken | str) -> PhoneFeatures:
    symbol = phone.symbol if isinstance(phone, PhoneToken) else phone
    if symbol in PHONE_FEATURES:
        return PHONE_FEATURES[symbol]
    if any(vowel in symbol for vowel in "aeiouyɯʌɛø"):
        return PhoneFeatures(symbol, "vowel")
    return PhoneFeatures(symbol, "consonant")


def base_substitution_cost(a: PhoneToken | str, b: PhoneToken | str) -> tuple[float, dict[str, float], str]:
    fa = get_phone_features(a)
    fb = get_phone_features(b)
    if fa.phone == fb.phone:
        return 0.0, {}, "match"
    if fa.category != fb.category:
        return 1.25, {"category": 1.25}, "cross-category confusion"

    penalties: dict[str, float] = {}
    total = 0.0
    if fa.category in {"consonant", "semivowel"}:
        for name, weight in [("place", 0.25), ("manner", 0.35), ("voicing", 0.10), ("aspiration", 0.18), ("tenseness", 0.22)]:
            if getattr(fa, name) != getattr(fb, name):
                penalties[name] = weight
                total += weight
    else:
        for name, weight in [("height", 0.28), ("backness", 0.28), ("rounding", 0.18)]:
            if getattr(fa, name) != getattr(fb, name):
                penalties[name] = weight
                total += weight

    label = "feature-similar substitution" if total <= 0.45 else "substitution"
    return min(total, 1.4), penalties, label


def language_profile_adjustment(a: PhoneToken | str, b: PhoneToken | str, profile: str = "ru") -> float:
    sa = a.symbol if isinstance(a, PhoneToken) else a
    sb = b.symbol if isinstance(b, PhoneToken) else b
    pair = frozenset({sa, sb})
    if pair in PROFILE_PAIR_OVERRIDES.get(profile, {}):
        return PROFILE_PAIR_OVERRIDES[profile][pair]
    return PROFILE_PAIR_OVERRIDES.get("default", {}).get(pair, 0.0)


def substitution_cost(a: PhoneToken | str, b: PhoneToken | str, context: dict | None = None, profile: str = "ru") -> tuple[float, str, dict[str, float], str]:
    cost, penalties, label = base_substitution_cost(a, b)
    cost = max(0.0, cost + language_profile_adjustment(a, b, profile=profile))

    left = a if isinstance(a, PhoneToken) else PhoneToken(str(a), "consonant")
    right = b if isinstance(b, PhoneToken) else PhoneToken(str(b), "consonant")

    if left.syllable_position == "coda" and right.category == "consonant" and right.symbol in {"k̚", "t̚", "p̚"}:
        cost = min(cost, 0.35)
        label = "coda neutralization"

    if {"aspiration", "tenseness"} & penalties.keys() and cost <= 0.35:
        label = "laryngeal contrast confusion"
    elif left.category == "vowel" and right.category == "vowel" and cost <= 0.4:
        label = "vowel confusion"
    elif left.symbol in {"ɾ", "l"} or right.symbol in {"ɾ", "l"}:
        label = "liquid realization issue"

    return cost, label, penalties, f"{left.symbol} -> {right.symbol}"


def insertion_cost(phone: PhoneToken | str, context: dict | None = None, profile: str = "ru") -> tuple[float, str]:
    token = phone if isinstance(phone, PhoneToken) else PhoneToken(str(phone), "consonant")
    cost = 0.85 if token.category == "vowel" else 1.0
    label = "epenthetic vowel insertion" if token.category == "vowel" else "consonant insertion"
    if profile == "ru" and token.category == "vowel":
        cost = 0.7
    return cost, label


def deletion_cost(phone: PhoneToken | str, context: dict | None = None, profile: str = "ru") -> tuple[float, str]:
    token = phone if isinstance(phone, PhoneToken) else PhoneToken(str(phone), "consonant")
    if token.syllable_position == "coda" or token.symbol.endswith("̚") or token.symbol == "l":
        return 0.75, "coda deletion"
    if token.category == "vowel":
        return 1.0, "vowel deletion"
    return 1.05, "consonant deletion"
