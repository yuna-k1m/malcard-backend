"""억양(Prosody) 분석 파이프라인 단일 진입점 — lens-rule paradigm v3.

backend가 호출하는 단일 함수. 반환은 UI에 그대로 전달 가능한 dict:
    {
      "reference_text": str,
      "records": list[record + feedback_text],  # 5 rule outlier + NL 문장
      "summary_when_no_outlier": str | None,    # records=[] 일 때만
      "prosody_plot": dict,                      # MFCC CMVN path 위 F0 + 어절 경계
    }

의존성:
    core/ — F0/MFCC 추출, segmenter, rules, feedback (외부 라이브러리 사용)
    src/  — AudioToIPARecognizer (Wav2Vec2), forced alignment, Korean IPA 변환

backend가 추가로 install해야 하는 외부 라이브러리 (기존 backend엔 없음):
    gtts           - core/tts.py (google-cloud-texttospeech 대체, 무료 한국어 TTS)
    dtaidistance   - core/rules.py, core/aligner.py (lens-rule DTW)
    plotly         - core/mfcc.py 등 top-level import (analyze.py 경로엔 미사용이나
                     모듈 로드 시 import 발생 — figure 함수를 함수 내부 lazy import로
                     옮기면 plotly 제외 가능)

이미 backend가 사용 중일 라이브러리 (librosa, parselmouth, scipy, numpy 등)는 그대로.

config/thresholds.toml은 rule 평가 파라미터 (외부화). repo root에서 상대경로로 자동 로드.

reference_text contract:
    - 어절 단위 띄어쓰기 필수. EojeolSegmenter가 split()으로 어절 분리하므로,
      띄어쓰기 없으면 발화 전체가 어절 1개로 취급되어 어절 단위 rule
      (pitch_rising/falling/offset/elongation/slow)이 사실상 무력화된다.
    - 구두점(. · , ! ? 。)은 함수 내부에서 자동 제거하므로 그대로 전달해도 무방하다.
      반환의 "reference_text"는 원본 그대로 (UI 표시용), 내부 처리만 cleaned.

사용법:
    from pronunciation_backend_pipeline import get_prosody_input
    from analyze import analyze

    prosody_input = get_prosody_input(audio_path, reference_text)
    result = analyze(prosody_input)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from pronunciation_backend_pipeline import get_default_recognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

if TYPE_CHECKING:
    from src.audio_to_ipa import AudioToIPARecognizer

_TTS_CACHE_DIR = Path("artifacts/tts_cache")


def _forced_align(wav_path: Path, text: str, recognizer: AudioToIPARecognizer):
    ipa_seq = pronunciation_to_ipa(text)
    candidate = PronunciationCandidate(pronunciation=text, ipa=ipa_seq, is_primary=True)
    recog = recognize_audio(recognizer, wav_path)
    vocab = recognizer.processor.tokenizer.get_vocab()
    blank_id = vocab[recognizer.processor.tokenizer.pad_token]
    return force_align_candidate(
        candidate,
        recog.logits,
        recog.frame_timestamps,
        label_to_id=vocab,
        blank_id=blank_id,
    )


def analyze(
    prosody_input: dict,
    *,
    recognizer: AudioToIPARecognizer | None = None,
    tts_cache_dir: str | Path | None = None,
) -> dict:
    """prosody_input dict → lens-rule 평가 + NL feedback + UI plot data 통합 dict.

    Args:
        prosody_input: `pronunciation_backend_pipeline.get_prosody_input()` 또는
                       `build_backend_payload()["prosody_input"]`의 반환값.
                       반드시 `audio_file_path` 키를 포함해야 한다
                       (build_backend_payload가 런타임에 주입하는 절대 경로).
        recognizer:    AudioToIPARecognizer 인스턴스. None이면
                       get_default_recognizer() 싱글톤. 같은 프로세스에서
                       음소 평가 파이프라인과 공유할 때 모델 중복 로드 회피.
        tts_cache_dir: TTS WAV 캐시 디렉토리. None이면 artifacts/tts_cache.

    Returns:
        {
          "reference_text": str,
          "records": [
            {
              "eojeol_idx": int,
              "rule_label": "pitch_rising_excess" | "pitch_falling_excess"
                          | "pitch_offset" | "syllable_elongation" | "eojeol_slow",
              "severity": "minor" | "major",
              "syllable_hint": str | None,
              "trigger_lens": str,
              "evidence_metrics": {...},  # rule별 dict
              "feedback_text": str        # deterministic 한국어 NL
            }, ...
          ],
          "summary_when_no_outlier": str,  # records=[] 일 때만 존재
          "prosody_plot": {
            "learner_f0_zscore":   list[float],
            "native_f0_zscore":    list[float],
            "learner_time_at_step": list[float],  # 초
            "eojeol_boundaries": [
              {"path_step": int, "label": str | None}, ...
            ]
          }
        }
    """
    # ── 파이프라인 개요 ──────────────────────────────────────────────────────
    # 1. prosody_input의 learner FA + reference text 파싱
    # 2. native(TTS) 합성 + 동일 forced alignment → 양쪽 음소 경계 확보
    # 3. F0/MFCC 추출
    # 4. 어절·음절 segmenter 인스턴스 → 경계 spans 계산
    # 5. lens-rule paradigm v3: rules.evaluate → list[Record]
    # 6. feedback.build_payload → records[].feedback_text + (records=[] 시) praise
    # 7. mfcc.build_prosody_plot_data → MFCC CMVN path 위 F0 lookup + 어절 boundary
    # 8. 통합 dict 반환

    _rec = recognizer or get_default_recognizer()
    _cache_dir = Path(tts_cache_dir) if tts_cache_dir is not None else _TTS_CACHE_DIR

    # ── Step 1. prosody_input 파싱 ───────────────────────────────────────────
    text = prosody_input["reference_text"]
    learner_wav = Path(prosody_input["audio_file_path"])
    learner_segments = prosody_input["phoneme_segments"]

    # ── Step 2. native(TTS) + forced alignment ───────────────────────────────
    # parselmouth/dtaidistance가 torch보다 먼저 import되면 macOS arm64 segfault.
    # 모델(recognizer)은 이미 로드된 상태에서 core/* 지연 import한다.
    from core.f0_extractor import extract_f0
    from core.feedback import build_payload as build_feedback_payload
    from core.mfcc import build_prosody_plot_data, extract_mfcc
    from core.rules import evaluate as evaluate_rules
    from core.segmenter import EojeolSegmenter, SyllableSegmenter
    from core.tts import generate_tts

    native_wav = generate_tts(text, cache_dir=_cache_dir)
    native_fa = _forced_align(native_wav, text, _rec)
    native_segments = [asdict(seg) for seg in native_fa.segments]

    # ── Step 3. F0 / MFCC 추출 ───────────────────────────────────────────────
    native_f0 = extract_f0(native_wav)
    learner_f0 = extract_f0(learner_wav)
    native_mfcc = extract_mfcc(native_wav)
    learner_mfcc = extract_mfcc(learner_wav)

    # ── Step 4. Segmenter (어절·음절 경계) ───────────────────────────────────
    # 구두점은 pronunciation_to_ipa의 position 할당을 깨뜨리므로 사전 제거.
    clean_text = "".join(c for c in text if c not in ".·,!?。")
    positions = [t.syllable_position for t in pronunciation_to_ipa(clean_text).tokens]
    syllable_labels = [c for c in text if c.strip() and c not in ".·,!?。"]

    eojeol_seg = EojeolSegmenter(native_segments, learner_segments, positions, clean_text)
    syllable_seg = SyllableSegmenter(
        native_segments, learner_segments, positions, syllable_labels,
    )

    # ── Step 5. Rule 평가 (lens-rule paradigm v3) ────────────────────────────
    records = evaluate_rules(
        native_f0, learner_f0,
        eojeol_native_spans=eojeol_seg.native_spans(),
        eojeol_learner_spans=eojeol_seg.learner_spans(),
        eojeol_labels=eojeol_seg.labels(),
        syllable_native_spans=syllable_seg.native_spans(),
        syllable_learner_spans=syllable_seg.learner_spans(),
        syllable_labels=syllable_seg.labels(),
        eojeol_text=clean_text,
    )

    # ── Step 6. NL feedback 합성 ─────────────────────────────────────────────
    payload = build_feedback_payload(records, text)

    # ── Step 7. UI plot data (MFCC CMVN path 위 F0 + 어절 boundary) ─────────
    payload["prosody_plot"] = build_prosody_plot_data(
        learner_mfcc, native_mfcc, learner_f0, native_f0,
        eojeol_learner_spans=eojeol_seg.learner_spans(),
        eojeol_labels=eojeol_seg.labels(),
    )

    return payload


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/20260421_220712_176144/20260421_220712_176144.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pi = data["prosody_input"]
    if "audio_file_path" not in pi:
        pi["audio_file_path"] = str(
            Path(path).parent / data["artifact_bundle"]["audio_file_name"]
        )

    result = analyze(pi)
    print(json.dumps(result, indent=2, ensure_ascii=False))
