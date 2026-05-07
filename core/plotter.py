import matplotlib.pyplot as plt
import numpy as np
from core.comparator import SyllableComparison
from core.f0_extractor import extract_f0
from core.metrics import SyllableMetrics

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class ComparisonPlotter:
    NATIVE_COLOR = '#185FA5'
    LEARNER_COLOR = '#993C1D'

    def __init__(
        self,
        native_label: str = '원어민',
        learner_label: str = '학습자',
        threshold: float = 1,
    ):
        self.native_label = native_label
        self.learner_label = learner_label
        self.threshold = threshold

    def plot(
        self,
        comparisons: list[SyllableComparison],
        metrics: list[SyllableMetrics],
        title: str = '억양 비교',
        syllable_labels: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        if not comparisons:
            return

        n_frames = len(comparisons[0].native_f0)
        native_all = np.concatenate([c.native_f0 for c in comparisons])
        learner_all = np.concatenate([c.learner_f0 for c in comparisons])
        frames = np.arange(len(native_all))

        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.plot(frames, native_all, color=self.NATIVE_COLOR, linewidth=2.5, label=self.native_label)
        ax.plot(frames, learner_all, color=self.LEARNER_COLOR, linewidth=2.5, label=self.learner_label)

        for i, (c, m) in enumerate(zip(comparisons, metrics)):
            x_start = i * n_frames
            x_mid = x_start + n_frames // 2
            ax.axvline(x=x_start, color='gray', linestyle='--', linewidth=0.5)

            if np.isnan(m.rmse):
                rmse_text, color = 'N/A', 'gray'
            elif m.rmse > self.threshold:
                rmse_text, color = f'R:{m.rmse:.2f}', 'red'
            else:
                rmse_text, color = f'R:{m.rmse:.2f}', 'gray'

            ax.text(x_mid, 2.7, rmse_text, ha='center', fontsize=8, color=color)

            if not np.isnan(m.pearson):
                p_color = 'red' if m.pearson < 0.5 else 'gray'
                ax.text(x_mid, 2.3, f'P:{m.pearson:.2f}', ha='center', fontsize=7, color=p_color)

            if syllable_labels and i < len(syllable_labels):
                ax.text(x_mid, -3.2, syllable_labels[i], ha='center', fontsize=10, color=color)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('리샘플 프레임')
        ax.set_ylabel('음높이 (z-score)')
        ax.set_ylim(-3.5, 3)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        return fig

    def plot_raw_f0(
        self,
        native_path: str,
        learner_path: str,
        title: str = 'Normalized Pitch 비교 (Hz)',
        save_path: str | None = None,
    ) -> None:
        """DTW 정렬 없이 각 wav의 실제 시간축 F0(Hz)를 나란히 그린다."""
        native_f0 = extract_f0(native_path)
        learner_f0 = extract_f0(learner_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        fig.suptitle(title, fontsize=14)

        for ax, f0_result, label, color in [
            (ax1, native_f0, self.native_label, self.NATIVE_COLOR),
            (ax2, learner_f0, self.learner_label, self.LEARNER_COLOR),
        ]:
            ax.plot(f0_result.times, f0_result.f0, color=color, linewidth=2, label=label)
            ax.set_ylabel('F0 (Normalized Hz)')
            ax.set_xlabel('시간 (초)')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig