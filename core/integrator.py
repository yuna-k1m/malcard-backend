"""Prosody inference 파이프라인.

prosody_input dict + KoreanDistributionStore → 어절별 판정 + 음절 drill-down.

입력:
    prosody_input: {
        "audio_file_path": str,
        "reference_text": str,
        "phoneme_segments": list[dict]  # src/ forced alignment 결과
    }
    store: KoreanDistributionStore

출력 schema:
    {
        "reference_text": str,
        "overall": { "verdict": str, "avg_mahalanobis": float },
        "eojeol_results": [ { "text", "idx", "prosody", "syllable_drilldown" } ]
    }
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.distribution import GaussianEojeolDistribution, KoreanDistributionStore
from core.eojeol_vector import DIM_NAMES, extract_eojeol_vector
from core.f0_extractor import F0Result, extract_f0
from core.syllable_utils import segments_to_syllable_boundaries
from src.korean_ipa import pronunciation_to_ipa

_VERDICT_THRESHOLDS = (1.5, 2.5)   # (natural/borderline, borderline/unnatural)
_N_CONTOUR_FRAMES = 50


# ── 어절 분할 ────────────────────────────────────────────────────────────────

def _split_segments_by_eojeol(
    segments: list[dict],
    eojeols: list[str],
) -> list[list[dict]] | None:
    """phoneme_segments를 어절별로 분할. 토큰 수 불일치 시 None."""
    token_counts = [len(pronunciation_to_ipa(ej).tokens) for ej in eojeols]
    if len(segments) != sum(token_counts):
        return None
    result, offset = [], 0
    for n in token_counts:
        result.append(segments[offset: offset + n])
        offset += n
    return result


# ── 음절 drill-down ──────────────────────────────────────────────────────────

def _resample_f0(f0_result: F0Result, t_start: float, t_end: float) -> np.ndarray:
    mask = (f0_result.times >= t_start) & (f0_result.times < t_end)
    f0 = f0_result.f0[mask]
    if len(f0) == 0:
        return np.zeros(_N_CONTOUR_FRAMES)
    src = np.linspace(0, _N_CONTOUR_FRAMES - 1, len(f0))
    return np.interp(np.arange(_N_CONTOUR_FRAMES, dtype=float), src, f0)


def _syllable_drilldown(
    f0_result: F0Result,
    syl_boundaries: list[tuple[float, float]],
    mean_contours: list[list[float]],
) -> list[dict]:
    """학습자 음절 F0 vs 한국인 평균 contour → RMSE per syllable."""
    drilldown = []
    for syl_idx, (t_start, t_end) in enumerate(syl_boundaries):
        learner_f0 = _resample_f0(f0_result, t_start, t_end)
        if syl_idx < len(mean_contours):
            korean_mean = np.array(mean_contours[syl_idx])
            rmse = float(np.sqrt(np.mean((learner_f0 - korean_mean) ** 2)))
        else:
            korean_mean = np.zeros(_N_CONTOUR_FRAMES)
            rmse = float("nan")
        drilldown.append({
            "idx": syl_idx,
            "learner_f0": learner_f0.tolist(),
            "korean_mean_f0": korean_mean.tolist(),
            "rmse": rmse,
        })
    return drilldown


# ── 전체 판정 ────────────────────────────────────────────────────────────────

def _verdict(avg_mahal: float) -> str:
    lo, hi = _VERDICT_THRESHOLDS
    if avg_mahal < lo:
        return "natural"
    if avg_mahal < hi:
        return "borderline"
    return "unnatural"


# ── 메인 진입점 ──────────────────────────────────────────────────────────────

def analyze_prosody(
    prosody_input: dict,
    store: KoreanDistributionStore,
) -> dict:
    """prosody_input → 어절별 Mahalanobis 판정 + outlier 음절 drill-down.

    Args:
        prosody_input: audio_file_path, reference_text, phoneme_segments 포함.
        store: build_korean_distribution.py로 생성된 분포 저장소.

    Returns:
        출력 schema dict.

    Raises:
        KeyError: reference_text가 분포에 없을 때.
    """
    text: str = prosody_input["reference_text"]
    wav_path = Path(prosody_input["audio_file_path"])
    segments: list[dict] = prosody_input["phoneme_segments"]

    dists = store.get(text)
    if dists is None:
        raise KeyError(f"reference_text not in distribution: {text!r}")

    eojeols = text.split()
    eojeol_segs = _split_segments_by_eojeol(segments, eojeols)
    if eojeol_segs is None:
        raise ValueError(
            f"phoneme_segments 수({len(segments)})가 예상 IPA 토큰 수와 불일치"
        )

    f0_result = extract_f0(wav_path)
    eojeol_texts = store.eojeol_texts(text)
    all_syl_contours = store.syllable_contours(text)  # [eojeol][syl] → [50 floats]

    eojeol_results = []
    mahal_values = []

    for i, (ej_text, ej_segs, dist) in enumerate(
        zip(eojeol_texts, eojeol_segs, dists)
    ):
        clean_ej = "".join(c for c in ej_text if c not in ".·,!?。")
        ej_positions = [t.syllable_position for t in pronunciation_to_ipa(clean_ej).tokens]
        syl_boundaries = segments_to_syllable_boundaries(ej_segs, ej_positions)
        t_start = ej_segs[0]["start_time"]
        t_end   = ej_segs[-1]["end_time"]

        vector = extract_eojeol_vector(f0_result, (t_start, t_end), syl_boundaries)
        mahal = dist.mahalanobis(vector)
        in_dist = dist.is_in_distribution(vector)
        z_scores = dist.per_dim_z(vector)
        labels = dist.classify(vector)

        mahal_values.append(mahal)

        prosody_entry = {
            "vector": vector.tolist(),
            "mahalanobis_distance": round(mahal, 4),
            "in_distribution": in_dist,
            "per_dim_z_scores": {k: round(v, 4) for k, v in z_scores.items()},
            "rule_labels": labels,
            "data_quality": {
                "covariance_mode": dist._mode,
                "n": dist._n,
            },
        }

        # outlier 어절에만 drill-down 포함
        drilldown = []
        if not in_dist:
            mean_contours = all_syl_contours[i] if i < len(all_syl_contours) else []
            drilldown = _syllable_drilldown(f0_result, syl_boundaries, mean_contours)

        eojeol_results.append({
            "text": ej_text,
            "idx": i,
            "boundary": [round(t_start, 4), round(t_end, 4)],
            "prosody": prosody_entry,
            "syllable_drilldown": drilldown,
        })

    avg_mahal = float(np.mean(mahal_values)) if mahal_values else float("nan")

    return {
        "reference_text": text,
        "overall": {
            "verdict": _verdict(avg_mahal),
            "avg_mahalanobis": round(avg_mahal, 4),
        },
        "eojeol_results": eojeol_results,
    }