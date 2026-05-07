"""억양(Prosody) 분석 파이프라인 단일 진입점.

의존성:
    core/   — F0 추출, segmental alignment, 메트릭 계산 (외부 의존 없음)
    src/    — AudioToIPARecognizer (Wav2Vec2), forced alignment, Korean IPA 변환

사용법:
    from pronunciation_backend_pipeline import get_prosody_input
    from analyze import analyze

    prosody_input = get_prosody_input(audio_path, reference_text)
    results = analyze(prosody_input)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from core.comparator import IntonationComparator
from core.f0_extractor import extract_f0
from core.metrics import compute_metrics, to_dict
from core.syllable_utils import segments_to_syllable_boundaries
from core.tts import generate_tts
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
) -> list[dict]:
    """prosody_input dict → 음절별 억양 분석 결과.

    Args:
        prosody_input: `pronunciation_backend_pipeline.get_prosody_input()` 또는
                       `build_backend_payload()["prosody_input"]`의 반환값.
                       반드시 `audio_file_path` 키를 포함해야 한다
                       (build_backend_payload가 런타임에 주입하는 절대 경로).
        recognizer:    AudioToIPARecognizer 인스턴스. None이면
                       get_default_recognizer() 싱글톤을 사용한다.
                       팀원 파이프라인과 같은 프로세스에서 호출할 때는
                       None으로 두어 모델 중복 로드를 방지한다.
        tts_cache_dir: TTS WAV 캐시 디렉토리 경로. None이면 기본값
                       `artifacts/tts_cache`를 사용한다.
                       프로덕션에서는 절대 경로 또는 공유 스토리지 경로를 전달하라.

    Returns:
        음절별 dict 리스트. 음절 수 = min(native 음절 수, learner 음절 수).
        모든 값은 JSON 직렬화 가능 (NaN → None).

        각 dict 필드:
            syllable_idx        (int)          음절 인덱스 (0부터)
            syllable_label      (str)          해당 음절의 한글 문자 (reference_text 기준)
            native_start        (float)        원어민 오디오 내 음절 시작 시각 (초)
            learner_start       (float)        학습자 오디오 내 음절 시작 시각 (초)
            native_f0           (list[float])  50프레임 z-score F0, 무성=0 (원어민 TTS)
            learner_f0          (list[float])  50프레임 z-score F0, 무성=0 (학습자)
            joint_voiced_mask   (list[bool])   두 화자 모두 유성인 프레임.
                                               False 구간은 프런트엔드에서 반투명/점선 처리
                                               등으로 "비교 불가 구간"임을 표시하는 데 활용.
            native_duration     (float)        원어민 음절 지속시간 (초)
            learner_duration    (float)        학습자 음절 지속시간 (초)
            rmse                (float|None)   F0 RMSE (z-score 단위); 유성 프레임 없으면 None
            pearson             (float|None)   억양 흐름 유사도 -1~1; 유성 프레임 < 5이면 None
            slope_diff          (float|None)   피치 변화율 차이 native-learner; 유성 프레임 < 3이면 None
            voiced_frame_count  (int)          유효 유성 프레임 수
            duration_ratio      (float|None)   learner/native 지속시간 비율; native=0이면 None
    """

    # ── pipeline 개요 ────────────────────────────────────────────────────────
    # 음소분석 파이프라인의 prosody_input은 learner 정보만 담고 있어
    # 억양 비교를 위한 native 기준값이 없다.
    # 이를 보완하기 위해 reference text로 TTS를 합성하고,
    # learner에 적용한 것과 동일한 forced alignment를 native에도 수행하여
    # 양쪽의 음소 경계를 확보한다.
    # 이 때문에 위에 _forced_align 함수는 억양분석 모듈(core)이 아닌
    # 음소분석 모듈(src)에 의존한다.
    #
    # 음소 경계 → 음절 경계로 변환한 뒤,
    # z-score 정규화(화자 간 음역대 차이 제거) + segmental alignment(음절 인덱스 1:1 매핑)로
    # native와 learner를 동일한 기준 위에 놓고 음절 단위로 비교한다.
    #
    # 음절별 비교 지표 3가지:
    #   RMSE        — F0 절대 차이. 억양이 얼마나 크게 벗어났는지
    #   Pearson     — 억양 흐름 유사도(-1~1). 올라가고 내려가는 방향이 같은지
    #   slope_diff  — 피치 변화율 차이(native - learner). 상승/하강 강도 비교
    #
    # 시각화와 API 전달에 필요한 모든 feature를 JSON 직렬화 가능한
    # list[dict] 형태로 반환한다.

    _rec = recognizer or get_default_recognizer()
    _cache_dir = Path(tts_cache_dir) if tts_cache_dir is not None else _TTS_CACHE_DIR

    # ── Step 1. prosody_input 파싱 ───────────────────────────────────────────
    # 음소분석 파이프라인이 런타임에 주입한 learner wav 절대 경로와
    # 강제 정렬 결과(phoneme_segments)를 읽는다.
    text = prosody_input["reference_text"]
    learner_wav = Path(prosody_input["audio_file_path"])
    learner_segments = prosody_input["phoneme_segments"]

    # ── Step 2. native(TTS) 생성 + forced alignment ──────────────────────────
    # TTS로 원어민 기준 오디오 합성 → Wav2Vec2 CTC로 음소별 시간 경계 추출
    # TTS 결과는 (text, voice, speed) 해시 기반으로 캐시됨 (재호출 비용 없음)
    native_wav = generate_tts(text, cache_dir=_cache_dir)
    native_fa = _forced_align(native_wav, text, _rec)
    native_segments = [asdict(seg) for seg in native_fa.segments]

    # ── Step 3. F0 추출 + z-score 정규화 ────────────────────────────────────
    # parselmouth(Praat)로 각 wav의 피치 곡선 추출
    # 화자 간 음역대 차이를 제거하기 위해 z-score 정규화 적용
    native_f0 = extract_f0(native_wav)
    learner_f0 = extract_f0(learner_wav)

    # ── Step 4. 음소 segments → 음절 경계 변환 ──────────────────────────────
    # 각 음절의 (t_start, t_end) 를 확보. nucleus(모음) 기준으로 경계를 묶는다.
    native_boundaries = segments_to_syllable_boundaries(native_segments)
    learner_boundaries = segments_to_syllable_boundaries(learner_segments)

    # ── Step 5. Segmental Alignment — 음절별 F0 비교 ─────────────────────────
    # 음절 인덱스 1:1 매핑. 각 음절을 50프레임으로 리샘플 후 비교.
    # zip 기준이므로 음절 수 불일치 시 min(native, learner) 개수로 맞춰진다.
    comparisons = IntonationComparator().compare(
        native_f0, learner_f0,
        native_boundaries=native_boundaries,
        learner_boundaries=learner_boundaries,
    )

    # ── Step 6. 멀티 메트릭 계산 + 직렬화 ───────────────────────────────────
    # 음절별 RMSE / Pearson / slope_diff / duration_ratio 산출
    # NaN → None 변환으로 JSON 직렬화 보장
    metrics = compute_metrics(comparisons)
    result = to_dict(comparisons, metrics)

    # ── Step 7. 시각화 보조 필드 주입 ───────────────────────────────────────
    # syllable_label : reference_text → 공백·구두점 제거 → 음절 단위 한글 문자
    # native_start   : 원어민 오디오 내 음절 시작 시각 (초) — 시간축 배치용
    # learner_start  : 학습자 오디오 내 음절 시작 시각 (초) — 시간축 배치용
    syllable_labels = [c for c in text if c.strip() and c not in ".·,!?。"]
    for item in result:
        idx = item["syllable_idx"]
        item["syllable_label"] = syllable_labels[idx]
        item["native_start"] = native_boundaries[idx][0]
        item["learner_start"] = learner_boundaries[idx][0]

    return result


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/20260421_220712_176144/20260421_220712_176144.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pi = data["prosody_input"]
    # audio_file_path는 런타임에 주입되므로 저장된 JSON에는 없을 수 있다.
    # 없으면 artifact_bundle에서 경로를 복원한다.
    if "audio_file_path" not in pi:
        pi["audio_file_path"] = str(
            Path(path).parent / data["artifact_bundle"]["audio_file_name"]
        )

    results = analyze(pi)
    print(json.dumps(results, indent=2, ensure_ascii=False))