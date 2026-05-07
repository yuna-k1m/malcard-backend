from __future__ import annotations

"""Build reference pronunciation candidates from Korean text."""

from dataclasses import dataclass

from src.korean_ipa import normalize_korean_text, pronunciation_to_ipa
from src.types import IPASequence, PronunciationCandidate, PronunciationReference


ONSETS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
VOWELS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
CODAS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

TENSE_MAP = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
ASPIRATE_MAP = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ", "ㅅ": "ㅆ"}
NASAL_CODA_MAP = {"ㄱ": "ㅇ", "ㄲ": "ㅇ", "ㅋ": "ㅇ", "ㄳ": "ㅇ", "ㄺ": "ㅇ", "ㄷ": "ㄴ", "ㅅ": "ㄴ", "ㅆ": "ㄴ", "ㅈ": "ㄴ", "ㅊ": "ㄴ", "ㅌ": "ㄴ", "ㅎ": "ㄴ", "ㅂ": "ㅁ", "ㅍ": "ㅁ", "ㄼ": "ㅁ", "ㅄ": "ㅁ"}
LIAISON_CODA = {"ㄱ", "ㄲ", "ㄷ", "ㅂ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"}
PALATALIZE_VOWELS = {"ㅣ", "ㅑ", "ㅕ", "ㅛ", "ㅠ", "ㅒ", "ㅖ"}


@dataclass
class Syllable:
    onset: str
    vowel: str
    coda: str
    space_before: bool = False


def _decompose_syllable(char: str) -> Syllable | None:
    if not ("가" <= char <= "힣"):
        return None
    code = ord(char) - ord("가")
    return Syllable(ONSETS[code // 588], VOWELS[(code % 588) // 28], CODAS[code % 28])


def _compose_syllable(syllable: Syllable) -> str:
    code = ord("가") + ONSETS.index(syllable.onset) * 588 + VOWELS.index(syllable.vowel) * 28 + CODAS.index(syllable.coda)
    return chr(code)


def _parse_text(text: str) -> list[Syllable | str]:
    items: list[Syllable | str] = []
    pending_space = False
    for char in text:
        if char.isspace():
            pending_space = True
            continue
        syllable = _decompose_syllable(char)
        if syllable is None:
            if pending_space and items and items[-1] != " ":
                items.append(" ")
            items.append(char)
            pending_space = False
            continue
        syllable.space_before = pending_space
        items.append(syllable)
        pending_space = False
    return items


def _iter_adjacent_syllables(items: list[Syllable | str]) -> list[tuple[int, int]]:
    indices = [i for i, item in enumerate(items) if isinstance(item, Syllable)]
    return list(zip(indices, indices[1:]))


def _clone_items(items: list[Syllable | str]) -> list[Syllable | str]:
    return [Syllable(item.onset, item.vowel, item.coda, item.space_before) if isinstance(item, Syllable) else item for item in items]


def _apply_phonological_rules(items: list[Syllable | str], *, aggressive: bool) -> list[Syllable | str]:
    items = _clone_items(items)
    for left_index, right_index in _iter_adjacent_syllables(items):
        left = items[left_index]
        right = items[right_index]
        assert isinstance(left, Syllable) and isinstance(right, Syllable)

        if left.coda in {"ㄷ", "ㅌ"} and right.onset == "ㅇ" and right.vowel in PALATALIZE_VOWELS:
            right.onset = "ㅈ" if left.coda == "ㄷ" else "ㅊ"
            left.coda = ""

        if left.coda in {"ㅎ", "ㄶ", "ㅀ"} and right.onset in ASPIRATE_MAP:
            right.onset = ASPIRATE_MAP[right.onset]
            left.coda = {"ㄶ": "ㄴ", "ㅀ": "ㄹ"}.get(left.coda, "")

        if left.coda in NASAL_CODA_MAP and right.onset in {"ㄴ", "ㅁ"}:
            left.coda = NASAL_CODA_MAP[left.coda]

        if (left.coda == "ㄴ" and right.onset == "ㄹ") or (left.coda == "ㄹ" and right.onset == "ㄴ"):
            left.coda = "ㄹ"
            right.onset = "ㄹ"

        if left.coda and right.onset in TENSE_MAP:
            if left.coda == "ㄺ" and right.onset == "ㄱ":
                left.coda = "ㄹ"
                right.onset = "ㄲ"
            elif left.coda not in {"ㄴ", "ㄹ", "ㅁ", "ㅇ"}:
                right.onset = TENSE_MAP[right.onset]

        if aggressive and left.coda in LIAISON_CODA and right.onset == "ㅇ":
            move = left.coda
            if move in {"ㄱ", "ㄲ", "ㅋ"}:
                right.onset = "ㄱ"
            elif move in {"ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ"}:
                right.onset = "ㄷ"
            elif move in {"ㅂ", "ㅍ"}:
                right.onset = "ㅂ"
            else:
                right.onset = move
            left.coda = ""
    return items


def _items_to_pronunciation(items: list[Syllable | str]) -> str:
    chars: list[str] = []
    for item in items:
        if item == " ":
            if chars and chars[-1] != " ":
                chars.append(" ")
            continue
        if isinstance(item, Syllable):
            if item.space_before and chars and chars[-1] != " ":
                chars.append(" ")
            chars.append(_compose_syllable(item))
        else:
            chars.append(item)
    return "".join(chars).strip()


def text_to_pronunciation(text: str) -> PronunciationReference:
    normalized = normalize_korean_text(text)
    if not normalized:
        raise ValueError("정답 문장을 입력해 주세요.")

    items = _parse_text(normalized)
    conservative_pron = _items_to_pronunciation(_apply_phonological_rules(items, aggressive=False))
    surface_pron = _items_to_pronunciation(_apply_phonological_rules(items, aggressive=True))

    candidates: list[PronunciationCandidate] = []
    seen: set[str] = set()
    for pronunciation, notes, is_primary in [
        (conservative_pron, ["대표 표준 발음형"], True),
        (surface_pron, ["연음/표면 변이 후보"], False),
        (normalized, ["표기형 기반 후보"], False),
    ]:
        if pronunciation and pronunciation not in seen:
            seen.add(pronunciation)
            candidates.append(
                PronunciationCandidate(
                    pronunciation=pronunciation,
                    ipa=pronunciation_to_ipa(pronunciation),
                    notes=notes,
                    is_primary=is_primary,
                )
            )

    representative = next(candidate for candidate in candidates if candidate.is_primary)
    return PronunciationReference(
        original_text=text,
        normalized_text=normalized,
        representative_pronunciation=representative.pronunciation,
        representative_ipa=representative.ipa,
        candidates=candidates,
    )


def build_reference_candidates(text: str) -> list[IPASequence]:
    return [candidate.ipa for candidate in text_to_pronunciation(text).candidates]
