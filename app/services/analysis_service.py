from app.adapters.phoneme_adapter import run_phoneme_analysis

def analyze_phoneme(
    audio_path: str,
    reference_text: str,
    profile: str = "ru",
    save_artifacts: bool = True
) -> dict:
    return run_phoneme_analysis(audio_path, reference_text, profile, save_artifacts)
