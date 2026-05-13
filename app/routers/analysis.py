import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.services.analysis_service import analyze_phoneme, analyze_full
from app.adapters.prosody_adapter import run_prosody_analysis

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

@router.post("/prosody")
async def prosody(body: dict):
    try:
        prosody_input = body.get("prosody_input", {})
        result = run_prosody_analysis(prosody_input)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={"code": "PIPELINE_ERROR", "message": str(e)})

@router.post("/full")
async def full(
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
        result = analyze_full(tmp_path, reference_text, profile, save_artifacts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "PIPELINE_ERROR", "message": str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
@router.post("/feedback")
async def feedback(body: dict):
    try:
        llm_feedback_input = body.get("llm_feedback_input", {})
        from app.adapters.feedback_adapter import run_feedback
        result = run_feedback(llm_feedback_input)
        return {"feedback": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "PIPELINE_ERROR", "message": str(e)})
