"""어절 단위 Gaussian 분포 fitting + Mahalanobis 판정 + rule-based 분류."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.eojeol_vector import DIM_NAMES, _N_DIMS

_MIN_SAMPLES_FULL_COV = 12  # 이 미만이면 diagonal covariance fallback


@dataclass
class DataQuality:
    covariance_mode: str   # "full_shrunk" | "diagonal"
    sample_count: int
    warning: str | None = None


class GaussianEojeolDistribution:
    """어절 하나에 대한 한국인 화자 분포 모델.

    fit() 후 mahalanobis(), per_dim_z(), classify()를 순서대로 호출.
    """

    mean_: np.ndarray      # (12,)
    std_: np.ndarray       # (12,)  per-dim std (z-score용)
    cov_inv_: np.ndarray   # (12, 12)
    _mode: str
    _n: int

    def fit(self, vectors: np.ndarray) -> None:
        """분포 추정.

        Args:
            vectors: shape (N, 12) float64. N < 12이면 diagonal fallback.
        """
        if vectors.ndim != 2 or vectors.shape[1] != _N_DIMS:
            raise ValueError(f"vectors shape must be (N, {_N_DIMS}), got {vectors.shape}")

        self._n = len(vectors)
        self.mean_ = np.mean(vectors, axis=0)
        self.std_ = np.std(vectors, axis=0, ddof=1)
        self.std_ = np.where(self.std_ < 1e-8, 1e-8, self.std_)

        if self._n < _MIN_SAMPLES_FULL_COV:
            self.cov_inv_ = np.diag(1.0 / (self.std_ ** 2))
            self._mode = "diagonal"
        else:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(vectors)
            try:
                self.cov_inv_ = np.linalg.inv(lw.covariance_)
                self._mode = "full_shrunk"
            except np.linalg.LinAlgError:
                self.cov_inv_ = np.diag(1.0 / (self.std_ ** 2))
                self._mode = "diagonal"

    def mahalanobis(self, v: np.ndarray) -> float:
        """Mahalanobis distance (scalar, ≥ 0)."""
        d = v - self.mean_
        dist_sq = float(d @ self.cov_inv_ @ d)
        return float(np.sqrt(max(dist_sq, 0.0)))

    def per_dim_z(self, v: np.ndarray) -> dict[str, float]:
        """각 dim의 z-score (분포 mean/std 기준)."""
        z = (v - self.mean_) / self.std_
        return {name: float(z[i]) for i, name in enumerate(DIM_NAMES)}

    def is_in_distribution(self, v: np.ndarray, threshold: float = 2.5) -> bool:
        return self.mahalanobis(v) <= threshold

    def classify(self, v: np.ndarray) -> list[str]:
        """per_dim_z 기반 rule-mapping으로 오류 카테고리 목록 반환.

        Returns:
            감지된 오류 label 리스트. 정상이면 빈 리스트.
        """
        z = self.per_dim_z(v)
        labels: list[str] = []
        if z["f0_slope"] > 2:
            labels.append("rising 과도")
        if z["f0_slope"] < -2:
            labels.append("falling 과도")
        if abs(z["f0_slope"]) < 0.5 and z["f0_range"] < -2:
            labels.append("억양 평탄")
        if z["last_syl_ratio"] > 2:
            labels.append("마지막 음절 elongation")
        if z["duration"] > 2:
            labels.append("어절 전체 느림")
        if z["voiced_ratio"] < -2:
            labels.append("voiced 비율 낮음")
        return labels

    def data_quality(self) -> DataQuality:
        warning = None
        if self._n < _MIN_SAMPLES_FULL_COV:
            warning = f"분포 sample 부족 ({self._n}개) — 결과 신뢰도 낮음"
        return DataQuality(
            covariance_mode=self._mode,
            sample_count=self._n,
            warning=warning,
        )

    def to_dict(self) -> dict:
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "cov_inv": self.cov_inv_.tolist(),
            "mode": self._mode,
            "n": self._n,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaussianEojeolDistribution":
        obj = cls.__new__(cls)
        obj.mean_ = np.array(d["mean"])
        obj.std_ = np.array(d["std"])
        obj.cov_inv_ = np.array(d["cov_inv"])
        obj._mode = d["mode"]
        obj._n = d["n"]
        return obj


class KoreanDistributionStore:
    """저장된 분포 JSON 로드 + 문장별 조회 인터페이스."""

    def __init__(self, path: str | Path) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._store: dict[str, list[GaussianEojeolDistribution]] = {}
        self._eojeol_texts: dict[str, list[str]] = {}
        # [text][eojeol_idx] → list of (50,) contours, one per syllable
        self._syllable_contours: dict[str, list[list[list[float]]]] = {}
        for text, entry in data["sentences"].items():
            self._store[text] = [
                GaussianEojeolDistribution.from_dict(e) for e in entry["eojeols"]
            ]
            self._eojeol_texts[text] = [e["text"] for e in entry["eojeols"]]
            self._syllable_contours[text] = [
                e.get("syllable_contours", []) for e in entry["eojeols"]
            ]

    def get(self, text: str) -> list[GaussianEojeolDistribution] | None:
        """문장 전체 텍스트로 어절별 분포 리스트 조회."""
        return self._store.get(text)

    def eojeol_texts(self, text: str) -> list[str]:
        return self._eojeol_texts.get(text, [])

    def syllable_contours(self, text: str) -> list[list[list[float]]]:
        """어절별 음절 평균 contour 조회. [[syl0_50frames], [syl1_50frames], ...]."""
        return self._syllable_contours.get(text, [])

    def texts(self) -> list[str]:
        return list(self._store.keys())