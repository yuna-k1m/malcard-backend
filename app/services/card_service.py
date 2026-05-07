from app.adapters.card_adapter import load_cards, find_card

def get_cards(type: str | None = None, limit: int = 20, offset: int = 0) -> dict:
    cards = load_cards()
    if type:
        cards = [c for c in cards if c.get("category") == type]
    total = len(cards)
    items = cards[offset:offset + limit]
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset
    }

def get_card(card_id: str) -> dict | None:
    return find_card(card_id)