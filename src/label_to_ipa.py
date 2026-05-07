from __future__ import annotations

"""Convert recognizer label vocabulary into the evaluator's IPA space."""

from src.ipa_utils import build_ipa_sequence
from src.types import PhoneToken, IPASequence


LABEL_TO_IPA = {
    "A": "a",
    "E": "e",
    "EO": "ʌ",
    "EU": "ɯ",
    "I": "i",
    "O": "o",
    "U": "u",
    "iA": "ja",
    "iE": "je",
    "iEO": "jʌ",
    "iO": "jo",
    "iU": "ju",
    "euI": "ɰi",
    "oA": "wa",
    "oE": "wɛ",
    "uEO": "wʌ",
    "uI": "wi",
    "G": "k",
    "GG": "k͈",
    "Kh": "kʰ",
    "D": "t",
    "DD": "t͈",
    "Th": "tʰ",
    "B": "p",
    "BB": "p͈",
    "Ph": "pʰ",
    "J": "tɕ",
    "JJ": "tɕ͈",
    "CHh": "tɕʰ",
    "S": "s",
    "SS": "s͈",
    "H": "h",
    "N": "n",
    "M": "m",
    "NG": "ŋ",
    "R": "ɾ",
    "L": "l",
    "k": "k̚",
    "t": "t̚",
    "p": "p̚",
    "|": " ",
}


PALATALIZING_VOWEL_LABELS = {"I", "iA", "iE", "iEO", "iO", "iU"}
IPA_TO_LABEL = {
    "a": "A",
    "e": "E",
    "ɛ": "E",
    "ʌ": "EO",
    "ɯ": "EU",
    "i": "I",
    "o": "O",
    "u": "U",
    "ja": "iA",
    "je": "iE",
    "jɛ": "iE",
    "jʌ": "iEO",
    "jo": "iO",
    "ju": "iU",
    "ɰi": "euI",
    "wa": "oA",
    "wɛ": "oE",
    "wʌ": "uEO",
    "wi": "uI",
    "k": "G",
    "k͈": "GG",
    "kʰ": "Kh",
    "t": "D",
    "t͈": "DD",
    "tʰ": "Th",
    "p": "B",
    "p͈": "BB",
    "pʰ": "Ph",
    "tɕ": "J",
    "tɕ͈": "JJ",
    "tɕʰ": "CHh",
    "s": "S",
    "s͈": "SS",
    "ɕ": "S",
    "ɕ͈": "SS",
    "h": "H",
    "n": "N",
    "m": "M",
    "ŋ": "NG",
    "ɾ": "R",
    "l": "L",
    "k̚": "k",
    "t̚": "t",
    "p̚": "p",
}

# Some narrow Korean IPA vowels do not have a dedicated recognizer label.
# For forced alignment only, we map them to the closest available label in the
# model vocabulary while keeping the reference IPA itself unchanged elsewhere.
ALIGNMENT_LABEL_OVERRIDES = {
    "we": "oE",
}


def labels_to_ipa(labels: list[str]) -> IPASequence:
    filtered = [label for label in labels if label and label not in {"[PAD]", "[UNK]", "<s>", "</s>"}]
    ipa_tokens: list[str] = []
    for index, label in enumerate(filtered):
        next_label = filtered[index + 1] if index + 1 < len(filtered) else None
        if label == "S" and next_label in PALATALIZING_VOWEL_LABELS:
            ipa_tokens.append("ɕ")
            continue
        if label == "SS" and next_label in PALATALIZING_VOWEL_LABELS:
            ipa_tokens.append("ɕ͈")
            continue
        ipa_tokens.append(LABEL_TO_IPA.get(label, label))
    ipa_text = " ".join(token for token in ipa_tokens if token != " ")
    return build_ipa_sequence(ipa_text)


def decode_label_text(raw_label_text: str) -> tuple[list[str], IPASequence]:
    labels = [chunk for chunk in raw_label_text.split() if chunk]
    return labels, labels_to_ipa(labels)


def ipa_token_to_label(token: PhoneToken | str) -> str:
    symbol = token.symbol if isinstance(token, PhoneToken) else token
    if symbol in ALIGNMENT_LABEL_OVERRIDES:
        return ALIGNMENT_LABEL_OVERRIDES[symbol]
    if symbol not in IPA_TO_LABEL:
        raise ValueError(f"Unsupported IPA token for alignment: {symbol}")
    return IPA_TO_LABEL[symbol]


def ipa_tokens_to_labels(tokens: list[PhoneToken]) -> list[str]:
    return [ipa_token_to_label(token) for token in tokens]
