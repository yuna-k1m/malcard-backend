from app.adapters.phoneme_adapter import run_phoneme_analysis
from app.adapters.prosody_adapter import run_prosody_analysis

def analyze_phoneme(
    audio_path: str,
    reference_text: str,
    profile: str = "ru",
    save_artifacts: bool = True
) -> dict:
    return run_phoneme_analysis(audio_path, reference_text, profile, save_artifacts)

def analyze_full(
    audio_path: str,
    reference_text: str,
    profile: str = "ru",
    save_artifacts: bool = True
) -> dict:
    phoneme_result = run_phoneme_analysis(audio_path, reference_text, profile, save_artifacts)
    
    evaluation_status = phoneme_result.get("status", {}).get("evaluation_status", "discarded")
    
    if evaluation_status == "ready":
        prosody_input = phoneme_result.get("prosody_input", {})
        prosody_result = run_prosody_analysis(prosody_input)
        prosody_executed = True
        reason = "ready"
    else:
        prosody_result = []
        prosody_executed = False
        reason = evaluation_status

    return {
        "phoneme_result": phoneme_result,
        "prosody_result": prosody_result,
        "pipeline_state": {
            "prosody_executed": prosody_executed,
            "reason": reason
        }
    }