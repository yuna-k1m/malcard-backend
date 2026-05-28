from fastapi import APIRouter, HTTPException
from app.adapters.card_adapter import find_card

router = APIRouter()

@router.post("/check")
def check_puzzle(body: dict):
    puzzle_id = body.get("puzzle_id")
    user_answer = body.get("answer", [])

    if not puzzle_id or not user_answer:
        raise HTTPException(status_code=400, detail={"code": "INVALID_INPUT", "message": "puzzle_id와 answer가 필요해요."})

    # card_id 추출 (예: cafeteria_01_p1 -> cafeteria_01)
    card_id = "_".join(puzzle_id.split("_")[:-1])
    card = find_card(card_id)

    if card is None:
        raise HTTPException(status_code=404, detail={"code": "CARD_NOT_FOUND", "message": "카드를 찾을 수 없어요."})

    puzzle = next((p for p in card.get("puzzles", []) if p["id"] == puzzle_id), None)
    if puzzle is None:
        raise HTTPException(status_code=404, detail={"code": "PUZZLE_NOT_FOUND", "message": "퍼즐을 찾을 수 없어요."})

    correct_answer = puzzle["answer"]
    is_correct = user_answer == correct_answer

    wrong_indices = []
    if not is_correct:
        for i, (u, c) in enumerate(zip(user_answer, correct_answer)):
            if u != c:
                wrong_indices.append(i)
        if len(user_answer) != len(correct_answer):
            wrong_indices.append(-1)

    return {
        "puzzle_id": puzzle_id,
        "is_correct": is_correct,
        "wrong_indices": wrong_indices,
        "correct_answer": correct_answer if not is_correct else None
    }
