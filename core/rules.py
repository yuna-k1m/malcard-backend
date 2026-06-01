"""5 rule trigger 로직 — lens 출력 + DTW path → list[Record].

lens-rule paradigm v3:
- pitch_rising/falling_excess: DTW path 위 learner-time 슬라이딩 윈도우 → 부호별 묶음 record
- pitch_offset / syllable_elongation / eojeol_slow: 기존 lens metric 그대로

threshold + window 파라미터는 config/thresholds.toml에서 로드 (hardcode 금지).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from core.f0_extractor import F0Result
from core.features import delta_f0, interp_unvoiced, slice_signal
from core.record import Record, RuleLabel, Severity, sort_by_severity

_HANGUL_LO, _HANGUL_HI = 0xAC00, 0xD7A3
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.toml"
_SYLLABLE_OVERLAP_TH = 0.3  # 윈도우가 음절 시간의 30% 이상이면 그 음절 포함 (hardcode — micro)


def load_config(path: str | Path | None = None) -> dict:
    """config/thresholds.toml 전체 로드 — {thresholds, window} 반환."""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "rb") as f:
        return tomllib.load(f)


def _quantize(value: float, threshold: float) -> Severity:
    return "major" if abs(value) >= 2 * threshold else "minor"



def _mean_voiced(arr: np.ndarray, voiced: np.ndarray) -> float | None:
    if arr.size == 0 or not voiced.any():
        return None
    return float(arr[voiced].mean())


def _eojeol_syllable_ranges(eojeol_text: str) -> list[tuple[int, int]]:
    """어절별 (음절 start_idx, end_idx exclusive). EojeolSegmenter._group과 동일 매핑."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for w in eojeol_text.split():
        n = sum(1 for ch in w if _HANGUL_LO <= ord(ch) <= _HANGUL_HI)
        ranges.append((cursor, cursor + n))
        cursor += n
    return ranges


def _position_hint(pos: int, total: int, label: str) -> str:
    if total == 1:
        prefix = "유일 음절"
    elif pos == 0:
        prefix = "첫 음절"
    elif pos == total - 1:
        prefix = "마지막 음절"
    else:
        prefix = f"{pos + 1}번째 음절"
    return f"{prefix} ({label})" if label else prefix


def _window_to_syllables(
    w_start: float, w_end: float,
    syll_spans: list[tuple[float, float]],
    syll_labels: list[str],
) -> str | None:
    """윈도우 절대 시간 → overlap ≥ threshold인 음절들 hint. 없으면 None."""
    included: list[tuple[int, str]] = []
    for i, (s0, s1) in enumerate(syll_spans):
        dur = s1 - s0
        if dur <= 0:
            continue
        ovlp = max(0.0, min(w_end, s1) - max(w_start, s0))
        if ovlp / dur >= _SYLLABLE_OVERLAP_TH:
            label = syll_labels[i] if i < len(syll_labels) else ""
            included.append((i, label))
    if not included:
        return None
    n_total = len(syll_spans)
    if len(included) == 1:
        i, lbl = included[0]
        return _position_hint(i, n_total, lbl)
    first_i, _ = included[0]
    last_i, _ = included[-1]
    labels_concat = "".join(lbl for _, lbl in included if lbl)
    first_hint = _position_hint(first_i, n_total, "")
    last_hint = _position_hint(last_i, n_total, "")
    suffix = f" ({labels_concat})" if labels_concat else ""
    return f"{first_hint}~{last_hint}{suffix}"


def evaluate(
    native_f0: F0Result,
    learner_f0: F0Result,
    eojeol_native_spans: list[tuple[float, float]],
    eojeol_learner_spans: list[tuple[float, float]],
    eojeol_labels: list[str],
    syllable_native_spans: list[tuple[float, float]],
    syllable_learner_spans: list[tuple[float, float]],
    syllable_labels: list[str],
    eojeol_text: str,
    config_path: str | Path | None = None,
) -> list[Record]:
    """모든 어절에 대해 5 rule 평가 → flat list[Record], severity 내림차순."""
    cfg = load_config(config_path)
    th = cfg["thresholds"]
    win = cfg["window"]

    records: list[Record] = []
    n_delta = delta_f0(native_f0)
    l_delta = delta_f0(learner_f0)
    syll_ranges = _eojeol_syllable_ranges(eojeol_text)

    n_eojeols = min(len(eojeol_native_spans), len(eojeol_learner_spans))
    for idx in range(n_eojeols):
        n_t0, n_t1 = eojeol_native_spans[idx]
        l_t0, l_t1 = eojeol_learner_spans[idx]
        label = eojeol_labels[idx] if idx < len(eojeol_labels) else ""
        s0, s1 = syll_ranges[idx] if idx < len(syll_ranges) else (0, 0)
        s1_eff = min(s1, len(syllable_learner_spans))
        syll_spans_eo = syllable_learner_spans[s0:s1_eff]
        syll_labels_eo = syllable_labels[s0:s1_eff]

        records.extend(_rule_pitch_shape_windowed(
            idx, label, native_f0, learner_f0,
            n_delta, l_delta, n_t0, n_t1, l_t0, l_t1,
            syll_spans_eo, syll_labels_eo, th, win,
        ))

        rec = _rule_pitch_offset(
            idx, label, native_f0, learner_f0,
            n_t0, n_t1, l_t0, l_t1, th["pitch_offset"],
        )
        if rec:
            records.append(rec)

        rec = _rule_eojeol_slow(
            idx, label, n_t0, n_t1, l_t0, l_t1, th["eojeol_slow"],
        )
        if rec:
            records.append(rec)

        s1_native = min(s1, len(syllable_native_spans))
        if s0 < s1_native and s0 < s1_eff:
            rec = _rule_syllable_elongation(
                idx, label, s0, min(s1_native, s1_eff),
                syllable_native_spans, syllable_learner_spans, syllable_labels,
                th["syllable_elongation"],
            )
            if rec:
                records.append(rec)

    return sort_by_severity(records)


def _rule_pitch_shape_windowed(
    idx: int, label: str,
    native_f0: F0Result, learner_f0: F0Result,
    n_delta: np.ndarray, l_delta: np.ndarray,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    syll_spans_eo: list[tuple[float, float]],
    syll_labels_eo: list[str],
    th: dict[str, float],
    win: dict[str, float],
) -> list[Record]:
    """DTW path + learner-time 슬라이딩 윈도우. 부호별로 0~2 record."""
    n_d, n_v, _ = slice_signal(native_f0.times, n_delta, native_f0.voiced_mask, n_t0, n_t1)
    l_d, l_v, l_t = slice_signal(learner_f0.times, l_delta, learner_f0.voiced_mask, l_t0, l_t1)
    if len(n_d) < 3 or len(l_d) < 3:
        return []

    # DTW는 무성 보간 후 연속 contour에서. metric은 원래 voiced frame만 사용 (mean_voiced).
    n_interp = interp_unvoiced(n_d, n_v)
    l_interp = interp_unvoiced(l_d, l_v)

    from dtaidistance import dtw  # parselmouth/torch 순서 보호 (deferred import 패턴 유지)
    path = dtw.warping_path(n_interp, l_interp)
    # path 인덱스: (native_idx, learner_idx). learner idx에 매핑된 native idx 모음.
    n_idx_by_l: dict[int, list[int]] = {}
    for ni, li in path:
        n_idx_by_l.setdefault(li, []).append(ni)

    dur = l_t1 - l_t0
    if dur <= 0:
        return []
    size = win["size_ratio"] * dur
    stride = win["stride_ratio"] * dur
    if stride <= 0 or size <= 0 or size > dur:
        return []
    n_windows = max(1, int((dur - size) / stride) + 1)

    th_r = th["pitch_rising_excess"]
    th_f = th["pitch_falling_excess"]

    rising_wins: list[tuple[float, dict]] = []
    falling_wins: list[tuple[float, dict]] = []

    for k in range(n_windows):
        w_start_t = l_t0 + k * stride
        w_end_t = w_start_t + size
        l_mask = (l_t >= w_start_t) & (l_t < w_end_t)
        if not l_mask.any():
            continue
        li_indices = np.where(l_mask)[0]
        li_start, li_end = int(li_indices[0]), int(li_indices[-1])

        n_idx_set: list[int] = []
        for li in range(li_start, li_end + 1):
            n_idx_set.extend(n_idx_by_l.get(li, []))
        if not n_idx_set:
            continue
        ni_start, ni_end = min(n_idx_set), max(n_idx_set)

        l_mean = _mean_voiced(l_d[li_start:li_end + 1], l_v[li_start:li_end + 1])
        n_mean = _mean_voiced(n_d[ni_start:ni_end + 1], n_v[ni_start:ni_end + 1])
        if l_mean is None or n_mean is None:
            continue
        diff = l_mean - n_mean

        if not (diff > th_r or diff < -th_f):
            continue

        ratio_s = (w_start_t - l_t0) / dur
        ratio_e = (w_end_t - l_t0) / dur
        syll_str = _window_to_syllables(w_start_t, w_end_t, syll_spans_eo, syll_labels_eo) or ""
        win_info = {
            "learner_time_ratio": [round(ratio_s, 3), round(ratio_e, 3)],
            "delta_diff": round(diff, 4),
            "syllable": syll_str,
        }
        if diff > th_r:
            rising_wins.append((diff, win_info))
        else:
            falling_wins.append((diff, win_info))

    records: list[Record] = []
    if rising_wins:
        records.append(_make_shape_record(
            idx, label, "pitch_rising_excess", rising_wins, th_r,
            syll_spans_eo, syll_labels_eo,
        ))
    if falling_wins:
        records.append(_make_shape_record(
            idx, label, "pitch_falling_excess", falling_wins, th_f,
            syll_spans_eo, syll_labels_eo,
        ))
    return records


def _make_shape_record(
    idx: int, eojeol_label: str,
    rule: RuleLabel, wins: list[tuple[float, dict]], threshold: float,
    syll_spans_eo: list[tuple[float, float]],
    syll_labels_eo: list[str],
) -> Record:
    max_abs = max(abs(d) for d, _ in wins)
    windows = [info for _, info in wins]  # k(시간) 순서 — append 순서 유지

    syllable_hint: str | None = None
    if syll_spans_eo:
        eo_start = syll_spans_eo[0][0]
        eo_end = syll_spans_eo[-1][1]
        eo_dur = eo_end - eo_start
        if eo_dur > 0:
            min_r = min(w["learner_time_ratio"][0] for w in windows)
            max_r = max(w["learner_time_ratio"][1] for w in windows)
            union_start = eo_start + min_r * eo_dur
            union_end = eo_start + max_r * eo_dur
            syllable_hint = _window_to_syllables(
                union_start, union_end, syll_spans_eo, syll_labels_eo
            )

    return Record(
        eojeol_idx=idx,
        rule_label=rule,
        severity=_quantize(max_abs, threshold),
        trigger_lens="eojeol_dtw_delta_window",
        syllable_hint=syllable_hint,
        evidence_metrics={
            "eojeol_label": eojeol_label,
            "windows": windows,
        },
    )


def _rule_pitch_offset(
    idx: int, label: str,
    native_f0: F0Result, learner_f0: F0Result,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    threshold: float,
) -> Record | None:
    n_f, n_v, _ = slice_signal(native_f0.times, native_f0.f0, native_f0.voiced_mask, n_t0, n_t1)
    l_f, l_v, _ = slice_signal(learner_f0.times, learner_f0.f0, learner_f0.voiced_mask, l_t0, l_t1)
    n_mean = _mean_voiced(n_f, n_v)
    l_mean = _mean_voiced(l_f, l_v)
    if n_mean is None or l_mean is None:
        return None
    diff = l_mean - n_mean
    if abs(diff) < threshold:
        return None
    return Record(
        eojeol_idx=idx,
        rule_label="pitch_offset",
        severity=_quantize(diff, threshold),
        trigger_lens="f0_extractor",
        evidence_metrics={
            "eojeol_label": label,
            "learner_eojeol_z_mean": round(l_mean, 4),
            "native_eojeol_z_mean": round(n_mean, 4),
            "z_diff": round(diff, 4),
        },
    )


def _rule_eojeol_slow(
    idx: int, label: str,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    threshold: float,
) -> Record | None:
    n_dur = n_t1 - n_t0
    l_dur = l_t1 - l_t0
    if n_dur <= 0 or l_dur <= 0:
        return None
    ratio = l_dur / n_dur
    deviation = ratio - 1.0
    trigger = threshold - 1.0  # 예: threshold=1.4 → 이탈 ≥0.4면 trigger
    if abs(deviation) < trigger:
        return None
    return Record(
        eojeol_idx=idx,
        rule_label="eojeol_slow",
        severity=_quantize(deviation, trigger),
        trigger_lens="eojeol_noalign",
        evidence_metrics={
            "eojeol_label": label,
            "learner_eojeol_dur_sec": round(l_dur, 3),
            "native_eojeol_dur_sec": round(n_dur, 3),
            "duration_ratio": round(ratio, 3),
        },
    )


def _rule_syllable_elongation(
    idx: int, label: str,
    s0: int, s1: int,
    syll_native_spans: list[tuple[float, float]],
    syll_learner_spans: list[tuple[float, float]],
    syll_labels: list[str],
    threshold: float,
) -> Record | None:
    if s0 >= s1:
        return None
    max_ratio = 0.0
    max_pos = -1
    learner_dur = native_dur = 0.0
    for i in range(s0, s1):
        n_dur = syll_native_spans[i][1] - syll_native_spans[i][0]
        l_dur = syll_learner_spans[i][1] - syll_learner_spans[i][0]
        if n_dur <= 0:
            continue
        ratio = l_dur / n_dur
        if ratio > max_ratio:
            max_ratio, max_pos = ratio, i - s0
            learner_dur, native_dur = l_dur, n_dur
    if max_pos < 0 or max_ratio < threshold:
        return None
    syll_label = syll_labels[s0 + max_pos] if (s0 + max_pos) < len(syll_labels) else ""
    deviation = max_ratio - 1.0
    trigger = threshold - 1.0
    return Record(
        eojeol_idx=idx,
        rule_label="syllable_elongation",
        severity=_quantize(deviation, trigger),
        trigger_lens="syllable_noalign",
        syllable_hint=_position_hint(max_pos, s1 - s0, syll_label),
        evidence_metrics={
            "eojeol_label": label,
            "syllable_idx_in_eojeol": max_pos,
            "syllable_label": syll_label,
            "learner_duration_sec": round(learner_dur, 3),
            "native_duration_sec": round(native_dur, 3),
            "duration_ratio": round(max_ratio, 3),
        },
    )
