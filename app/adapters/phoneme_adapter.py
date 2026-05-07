import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from pronunciation_backend_pipeline import evaluate_pronunciation_file

def run_phoneme_analysis(
    audio_path: str,
    reference_text: str,
    profile: str = "ru",
    save_artifacts: bool = True
) -> dict:
    return evaluate_pronunciation_file(
        audio_path=audio_path,
        reference_text=reference_text,
        profile=profile,
        save_artifacts=save_artifacts
    )