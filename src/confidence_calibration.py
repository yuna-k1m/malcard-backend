from __future__ import annotations

"""Token-level confidence calibration for forced-alignment gates."""

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration_stats.json"
DEFAULT_TOKEN_MIN_COUNT = 30
RAW_LOW_CONFIDENCE_THRESHOLD = 0.20
RAW_VERY_LOW_CONFIDENCE_THRESHOLD = 0.05
CALIBRATED_LOW_QUANTILE = 0.25
CALIBRATED_VERY_LOW_QUANTILE = 0.10
CALIBRATED_LOW_FLOOR = 0.01
CALIBRATED_VERY_LOW_FLOOR = 0.001


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _artifact_status(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if isinstance(status, dict):
        return status.get("evaluation_status")
    if isinstance(status, str):
        return status
    return payload.get("evaluation_status")


def _artifact_alignment_gate_passed(payload: dict[str, Any]) -> bool | None:
    gates = payload.get("gates") or {}
    alignment_gate = gates.get("alignment_confidence_gate") or {}
    return alignment_gate.get("passed")


def _artifact_forced_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    alignment = payload.get("alignment") or {}
    forced = alignment.get("forced") or {}
    segments = forced.get("segments") or []
    return segments if isinstance(segments, list) else []


def collect_token_confidences_from_artifacts(
    artifacts_dir: str | Path,
    *,
    source_status: str = "ready",
    require_alignment_gate_passed: bool = True,
) -> dict[str, list[float]]:
    """Collect per-token alignment confidences from existing artifact JSON files."""

    confidences: dict[str, list[float]] = defaultdict(list)
    for path in sorted(Path(artifacts_dir).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if source_status and _artifact_status(payload) != source_status:
            continue
        if require_alignment_gate_passed and _artifact_alignment_gate_passed(payload) is not True:
            continue

        for segment in _artifact_forced_segments(payload):
            token = segment.get("token")
            confidence = segment.get("confidence")
            if token and isinstance(confidence, (int, float)):
                confidences[str(token)].append(float(confidence))
    return dict(confidences)


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        "median": float(median(values)) if values else 0.0,
        "p05": _quantile(values, 0.05),
        "p10": _quantile(values, 0.10),
        "p20": _quantile(values, 0.20),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "raw_very_low_rate": (
            sum(value < RAW_VERY_LOW_CONFIDENCE_THRESHOLD for value in values) / len(values)
            if values
            else 0.0
        ),
        "raw_low_rate": (
            sum(value < RAW_LOW_CONFIDENCE_THRESHOLD for value in values) / len(values)
            if values
            else 0.0
        ),
    }


def build_confidence_calibration_stats(
    artifacts_dir: str | Path,
    *,
    token_min_count: int = DEFAULT_TOKEN_MIN_COUNT,
    source_status: str = "ready",
    require_alignment_gate_passed: bool = True,
) -> dict[str, Any]:
    """Build a conservative token-level calibration table from artifact JSON files."""

    token_confidences = collect_token_confidences_from_artifacts(
        artifacts_dir,
        source_status=source_status,
        require_alignment_gate_passed=require_alignment_gate_passed,
    )
    all_confidences = [
        confidence
        for confidences in token_confidences.values()
        for confidence in confidences
    ]

    tokens: dict[str, dict[str, Any]] = {}
    for token, confidences in sorted(token_confidences.items()):
        if len(confidences) < token_min_count:
            continue

        stats = _distribution(confidences)
        p10 = float(stats["p10"])
        p25 = float(stats["p25"])
        stats["calibrated_very_low_threshold"] = max(
            CALIBRATED_VERY_LOW_FLOOR,
            min(RAW_VERY_LOW_CONFIDENCE_THRESHOLD, p10),
        )
        stats["calibrated_low_threshold"] = max(
            CALIBRATED_LOW_FLOOR,
            min(RAW_LOW_CONFIDENCE_THRESHOLD, p25),
        )
        tokens[token] = stats

    return {
        "schema_version": 1,
        "description": (
            "Token-level forced-alignment confidence calibration. Thresholds are "
            "only relaxed for tokens that are empirically low-confidence in ready, "
            "alignment-passed artifacts; global thresholds remain the fallback."
        ),
        "source": {
            "artifacts_dir": str(Path(artifacts_dir)),
            "source_status": source_status,
            "require_alignment_gate_passed": require_alignment_gate_passed,
            "artifact_token_count": len(all_confidences),
        },
        "parameters": {
            "token_min_count": token_min_count,
            "raw_low_confidence_threshold": RAW_LOW_CONFIDENCE_THRESHOLD,
            "raw_very_low_confidence_threshold": RAW_VERY_LOW_CONFIDENCE_THRESHOLD,
            "calibrated_low_quantile": CALIBRATED_LOW_QUANTILE,
            "calibrated_very_low_quantile": CALIBRATED_VERY_LOW_QUANTILE,
            "calibrated_low_floor": CALIBRATED_LOW_FLOOR,
            "calibrated_very_low_floor": CALIBRATED_VERY_LOW_FLOOR,
        },
        "global": _distribution(all_confidences),
        "tokens": tokens,
    }


@lru_cache(maxsize=8)
def _load_confidence_calibration_cached(path_text: str) -> dict[str, Any] | None:
    path = Path(path_text)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), dict):
        raise ValueError(f"Invalid confidence calibration file: {path}")
    payload = dict(payload)
    payload["_loaded_from"] = str(path)
    return payload


def load_confidence_calibration(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load calibration stats, returning None when the default file is absent."""

    calibration_path = Path(path) if path is not None else DEFAULT_CALIBRATION_PATH
    return _load_confidence_calibration_cached(str(calibration_path.resolve()))


def get_token_thresholds(
    calibration_stats: dict[str, Any] | None,
    token: str,
    *,
    default_low_threshold: float = RAW_LOW_CONFIDENCE_THRESHOLD,
    default_very_low_threshold: float = RAW_VERY_LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Return calibrated low/very-low thresholds for one token."""

    if not calibration_stats:
        return {
            "source": "raw_default",
            "count": None,
            "low_threshold": default_low_threshold,
            "very_low_threshold": default_very_low_threshold,
        }

    token_stats = (calibration_stats.get("tokens") or {}).get(token)
    if token_stats:
        return {
            "source": "token_calibration",
            "count": token_stats.get("count"),
            "low_threshold": float(token_stats.get("calibrated_low_threshold", default_low_threshold)),
            "very_low_threshold": float(token_stats.get("calibrated_very_low_threshold", default_very_low_threshold)),
        }

    return {
        "source": "calibration_global_fallback",
        "count": None,
        "low_threshold": default_low_threshold,
        "very_low_threshold": default_very_low_threshold,
    }


def calibration_debug_summary(calibration_stats: dict[str, Any] | None) -> dict[str, Any]:
    if not calibration_stats:
        return {
            "enabled": False,
        }
    return {
        "enabled": True,
        "schema_version": calibration_stats.get("schema_version"),
        "loaded_from": calibration_stats.get("_loaded_from"),
        "token_count": len(calibration_stats.get("tokens") or {}),
        "token_min_count": (calibration_stats.get("parameters") or {}).get("token_min_count"),
        "artifact_token_count": (calibration_stats.get("source") or {}).get("artifact_token_count"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build token-level confidence calibration stats.")
    parser.add_argument("artifacts_dir", nargs="?", default="artifacts")
    parser.add_argument("output_path", nargs="?", default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument("--token-min-count", type=int, default=DEFAULT_TOKEN_MIN_COUNT)
    args = parser.parse_args()

    stats = build_confidence_calibration_stats(
        args.artifacts_dir,
        token_min_count=args.token_min_count,
    )
    output_path = Path(args.output_path)
    output_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"wrote {output_path} "
        f"tokens={len(stats.get('tokens') or {})} "
        f"artifact_token_count={(stats.get('source') or {}).get('artifact_token_count')}"
    )


if __name__ == "__main__":
    main()
