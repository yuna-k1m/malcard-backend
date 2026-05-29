import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from analyze import analyze

def run_prosody_analysis(prosody_input: dict) -> list[dict]:
    return analyze(prosody_input)
