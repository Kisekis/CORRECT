from common.dataloader import DataLoader
import json
from collections import Counter
import matplotlib.pyplot as plt
import os
from .base_evaluator import Evaluator


class FeedbackEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

    def evaluate(self):

        total_failed = 0
        feedback_counts = Counter()

        for model_name, model_data in self.model_meta_data.items():
            for data in model_data["data"].values():
                if data.normal_patched_ret == 0:
                    total_failed += 1
                    if (
                        hasattr(data, "hard_patched_feedback")
                        and "k" in data.hard_patched_feedback
                        and data.hard_patched_feedback["k"] != 0
                        and data.hard_patched_feedback["k"] < 5
                    ):
                        feedback_counts[data.hard_patched_feedback["k"]] += 1

        plt.figure(figsize=(10, 6))

        x_values = range(6)
        y_values = [
            feedback_counts[x] / total_failed if total_failed > 0 else 0
            for x in x_values
        ]

        plt.plot(x_values, y_values, marker="o", linewidth=2, markersize=8)

        plt.title("Distribution of Feedback Scores for Failed Cases", fontsize=14)
        plt.xlabel("Feedback Score", fontsize=12)
        plt.ylabel("Proportion", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)

        for x, y in zip(x_values, y_values):
            plt.text(x, y, f"{y:.1%}", ha="center", va="bottom")

        plt.ylim(0, max(y_values) * 1.1)
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: "{:.1%}".format(y))
        )

        os.makedirs("metrics_plots", exist_ok=True)
        plt.savefig(
            "metrics_plots/feedback_distribution.pdf", bbox_inches="tight", dpi=300
        )
        plt.close()

        print(f"All cases: {total_failed}")
        print("Distribution of feedback scores:")
        for score in sorted(feedback_counts.keys()):
            proportion = (
                feedback_counts[score] / total_failed if total_failed > 0 else 0
            )
            print(f"Score {score}: {feedback_counts[score]} cases ({proportion:.1%})")
