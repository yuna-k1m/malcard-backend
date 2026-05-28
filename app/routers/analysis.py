import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.analysis_service import analyze_phoneme

router = APIRouter()

@router.post("/phoneme")
async def phoneme(
    audio: UploadFile = File(...),
    reference_text: str = Form(...),
    profile: str = Form(default="ru"),
    save_artifacts: bool = Form(default=True)
):
    tmp_path = None
    try:
        suffix = os.path.splitext(audio.filename)[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name
        result = analyze_phoneme(tmp_path, reference_text, profile, save_artifacts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "PIPELINE_ERROR", "message": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
