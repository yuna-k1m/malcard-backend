from fastapi import APIRouter, HTTPException, Query
from app.services.card_service import get_cards, get_card

router = APIRouter()

@router.get("")
def list_cards(
    type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    return get_cards(type=type, limit=limit, offset=offset)

@router.get("/{card_id}")
def detail_card(card_id: str):
    card = get_card(card_id)
    if card is None:
        raise HTTPException(status_code=404, detail={"code": "CARD_NOT_FOUND", "message": "카드를 찾을 수 없어요."})
    return card