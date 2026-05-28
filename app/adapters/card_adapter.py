import json
from pathlib import Path

CARDS_PATH = Path(__file__).parent.parent.parent / "cards.json"

def load_cards() -> list[dict]:
    with open(CARDS_PATH, encoding="utf-8") as f:
        return json.load(f)

def find_card(card_id: str) -> dict | None:
    cards = load_cards()
    for card in cards:
        if card.get("id") == card_id:
            return card
    return None