from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"ok": True, "service": "malcard-api", "version": "1.0.0"}