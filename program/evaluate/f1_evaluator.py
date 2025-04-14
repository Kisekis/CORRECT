from evaluate.base_evaluator import Evaluator
import matplotlib.pyplot as plt
import os
import math
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import matplotlib.font_manager as fm


DEEPSEEK_R_COLOR = "#5BB5AC"
DEEPSEEK_NR_COLOR = "#8bdad2"
QWEN_R_COLOR = "#D8B365"
QWEN_NR_COLOR = "#ebcb88"
LLAMA_R_COLOR = "#DE526C"
LLAMA_NR_COLOR = "#f37e94"
GPT_R_COLOR = "#82b0d2"


class F1Evaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.drop_data("r1-7b")

    def evaluate(self):

        results = {
            "no_context": {"accuracy": [], "precision": [], "recall": [], "f1": []},
            "context": {"accuracy": [], "precision": [], "recall": [], "f1": []},
            "normal": {"accuracy": [], "precision": [], "recall": [], "f1": []},
            "hard": {"accuracy": [], "precision": [], "recall": [], "f1": []},
        }
        model_params = []

        for model_name, model_info in self.model_meta_data.items():
            data = model_info["data"]
            model_params.append(model_info["param"])

            for eval_type in ["no_context", "context", "normal", "hard"]:
                tp, fp, tn, fn = 0, 0, 0, 0

                for result in data.values():
                    try:

                        if eval_type == "no_context":
                            vuln_ret = result.no_context_vuln_ret
                            patched_ret = result.no_context_patched_ret
                        elif eval_type == "context":
                            vuln_ret = result.context_vuln_ret
                            patched_ret = result.context_patched_ret
                        elif eval_type == "normal":
                            vuln_ret = result.normal_vuln_ret
                            patched_ret = result.normal_patched_ret
                        else:
                            vuln_ret = result.hard_vuln_ret
                            patched_ret = result.hard_patched_ret

                        if vuln_ret == 1:
                            tp += 1
                        elif vuln_ret == 0:

                            fn += 1

                        if patched_ret == 1:
                            fp += 1
                        elif patched_ret == 0:
                            tn += 1
                    except AttributeError as e:
                        print(
                            f"Warning: Skipping result, attribute not found: {str(e)}"
                        )
                        continue

                total = tp + fp + tn + fn
                if total == 0:
                    continue

                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                results[eval_type]["accuracy"].append(accuracy)
                results[eval_type]["precision"].append(precision)
                results[eval_type]["recall"].append(recall)
                results[eval_type]["f1"].append(f1)
        self.results = results

        try:
            os.makedirs("metrics_plots", exist_ok=True)

            for metric in ["precision", "recall", "f1"]:
                self.draw_all_mode(metric)
            self.draw_final()
        except Exception as e:
            print(f"Error: {str(e)}")

    def draw_single(self, mode, metric, legend_on=False, ax=None, show_y_label=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3.3))

        regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )

        model_indices = {
            model_name: idx
            for idx, model_name in enumerate(self.model_meta_data.keys())
        }

        llama_reasoning_data = {"params": [], "results": []}
        qwen_reasoning_data = {"params": [], "results": []}
        llama_non_reasoning_data = {"params": [], "results": []}
        qwen_non_reasoning_data = {"params": [], "results": []}

        for model_name, model_info in self.model_meta_data.items():
            param = model_info["param"]
            current_result = self.results[mode][metric][model_indices[model_name]]

            if model_name in ["r1-8b", "r1-70b"]:
                llama_reasoning_data["params"].append(param)
                llama_reasoning_data["results"].append(current_result)
            elif model_name in ["r1-7b", "r1-14b", "r1-32b"]:
                qwen_reasoning_data["params"].append(param)
                qwen_reasoning_data["results"].append(current_result)
            elif model_name in ["8b", "70b"]:
                llama_non_reasoning_data["params"].append(param)
                llama_non_reasoning_data["results"].append(current_result)
            elif model_name in ["7b", "14b", "32b"]:
                qwen_non_reasoning_data["params"].append(param)
                qwen_non_reasoning_data["results"].append(current_result)

        if metric == "f1" and mode in ["normal", "hard"]:

            all_params = []
            for model_info in self.model_meta_data.values():
                if model_info["param"] is not None:
                    all_params.append(model_info["param"])
            x_min, x_max = min(all_params), max(all_params)

            reasoning_x = []
            reasoning_y = []

            for data in [llama_reasoning_data, qwen_reasoning_data]:
                if data["params"]:
                    for param, result in zip(data["params"], data["results"]):
                        reasoning_x.append(param)
                        reasoning_y.append(result)

            if "r1" in self.model_meta_data:
                param = self.model_meta_data["r1"]["param"]
                current_result = self.results[mode][metric][model_indices["r1"]]
                reasoning_x.append(param)
                reasoning_y.append(current_result)

            if reasoning_x:

                sorted_indices = sorted(
                    range(len(reasoning_x)), key=lambda k: reasoning_x[k]
                )
                x_sorted = [reasoning_x[i] for i in sorted_indices]
                y_sorted = [reasoning_y[i] for i in sorted_indices]

                slope, intercept, _, _, _ = stats.linregress(
                    np.log10(x_sorted), y_sorted
                )

                x_smooth = np.geomspace(x_min - 6, x_max + 100, 100)
                y_trend = slope * np.log10(x_smooth) + intercept

                offset = 0.03
                ax.fill_between(
                    x_smooth,
                    y_trend - offset * 1.5,
                    y_trend + offset * 1.5,
                    alpha=0.06,
                    color="blue",
                    label="Reasoning trend",
                )

            non_reasoning_x = []
            non_reasoning_y = []

            for data in [llama_non_reasoning_data, qwen_non_reasoning_data]:
                if data["params"]:
                    for param, result in zip(data["params"], data["results"]):
                        non_reasoning_x.append(param)
                        non_reasoning_y.append(result)

            if "v3" in self.model_meta_data:
                param = self.model_meta_data["v3"]["param"]
                current_result = self.results[mode][metric][model_indices["v3"]]
                non_reasoning_x.append(param)
                non_reasoning_y.append(current_result)

            if non_reasoning_x:

                sorted_indices = sorted(
                    range(len(non_reasoning_x)), key=lambda k: non_reasoning_x[k]
                )
                x_sorted = [non_reasoning_x[i] for i in sorted_indices]
                y_sorted = [non_reasoning_y[i] for i in sorted_indices]

                slope, intercept, _, _, _ = stats.linregress(
                    np.log10(x_sorted), y_sorted
                )

                x_smooth = np.geomspace(x_min - 6, x_max + 100, 100)
                y_trend = slope * np.log10(x_smooth) + intercept

                offset = 0.03
                ax.fill_between(
                    x_smooth,
                    y_trend - offset * 1.5 + 0.005,
                    y_trend + offset * 1.5 + 0.04,
                    alpha=0.08,
                    color="purple",
                    label="Non-reasoning trend",
                )

        plot_groups = [
            (llama_reasoning_data, "-", LLAMA_R_COLOR, "o", "Llama (reasoning)"),
            (qwen_reasoning_data, "-", QWEN_R_COLOR, "s", "Qwen (reasoning)"),
            (
                llama_non_reasoning_data,
                "--",
                LLAMA_NR_COLOR,
                "o",
                "Llama (non-reasoning)",
            ),
            (qwen_non_reasoning_data, "--", QWEN_NR_COLOR, "s", "Qwen (non-reasoning)"),
        ]

        for data, linestyle, color, marker, label in plot_groups:
            if data["params"]:
                sorted_idx = sorted(
                    range(len(data["params"])), key=lambda k: data["params"][k]
                )
                sorted_params = [data["params"][i] for i in sorted_idx]
                sorted_results = [data["results"][i] for i in sorted_idx]
                ax.plot(
                    sorted_params,
                    sorted_results,
                    linestyle,
                    color=color,
                    linewidth=2,
                    marker=marker,
                    markersize=8,
                    label=label,
                )

        if "o3-mini-medium" in self.model_meta_data and metric == "f1":
            o3_idx = list(self.model_meta_data.keys()).index("o3-mini-medium")
            o3_value = self.results[mode]["f1"][o3_idx]
            x_lim = ax.get_xlim()
            ax.hlines(
                y=o3_value,
                xmin=x_lim[0],
                xmax=1000,
                colors=GPT_R_COLOR,
                linestyles="--",
                linewidth=2,
            )

        for model_name in ["r1", "v3"]:
            if model_name in self.model_meta_data:
                param = self.model_meta_data[model_name]["param"]
                current_result = self.results[mode][metric][model_indices[model_name]]
                color = DEEPSEEK_NR_COLOR if model_name == "v3" else DEEPSEEK_R_COLOR
                if model_name == "r1":
                    label = f"Deepseek-{model_name} (reasoning)"
                elif model_name == "v3":
                    label = f"Deepseek-{model_name} (non-reasoning)"

                ax.plot(
                    [param],
                    [current_result],
                    "*",
                    color=color,
                    markersize=12,
                    label=label,
                )

        if legend_on:

            first_legend = ax.legend(
                bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="x-small"
            )

            custom_lines = [
                Line2D([0], [0], color="gray", linestyle="-", linewidth=2),
                Line2D([0], [0], color="gray", linestyle="--", linewidth=2),
            ]

            ax.legend(
                handles=[*first_legend.get_lines(), *custom_lines],
                labels=[
                    *[l.get_label() for l in first_legend.get_lines()],
                    "With Reasoning",
                    "Without Reasoning",
                ],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize="x-small",
            )

        ax.grid(True, which="major", ls="-", alpha=0.2)

        ax.set_xscale("log")
        ax.set_xlabel("Model Size (B)")
        if show_y_label:
            ax.set_ylabel(f"{metric.capitalize()} Score")
        ax.set_title(f'{mode.replace("_", " ").title()}', fontsize=10, weight="bold")

        self._set_axis_properties(ax, mode, metric)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(regular_font)

        return ax

    def draw_all_mode(self, metric):

        fig = plt.figure(figsize=(11, 3))

        gs = fig.add_gridspec(2, 4, height_ratios=[5, 1])
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        modes = ["no_context", "context", "normal", "hard"]
        for i, mode in enumerate(modes):

            self.draw_single(
                mode, metric, legend_on=False, ax=axes[i], show_y_label=(i == 0)
            )

        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=LLAMA_R_COLOR,
                linestyle="-",
                marker="o",
                label="Llama (reasoning)",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color=LLAMA_NR_COLOR,
                linestyle="--",
                marker="o",
                label="Llama (non-reasoning)",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_R_COLOR,
                linestyle="-",
                marker="s",
                label="Qwen (reasoning)",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_NR_COLOR,
                linestyle="--",
                marker="s",
                label="Qwen (non-reasoning)",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_R_COLOR,
                marker="*",
                label="DeepSeek-R1",
                linestyle="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_NR_COLOR,
                marker="*",
                label="DeepSeek-V3",
                linestyle="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                color=GPT_R_COLOR,
                linestyle="--",
                label="o3-mini",
                markersize=8,
            ),
        ]

        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            ncol=7,
            fontsize="x-small",
            bbox_to_anchor=(0.5, 0.5),
            frameon=True,
        )

        plt.tight_layout()
        plt.savefig(
            f"metrics_plots/all_modes_{metric}.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _set_axis_properties(self, ax, mode, metric):

        all_params = []
        for model_name, model_info in self.model_meta_data.items():
            if model_info["param"] is not None and model_name != "qwq":
                all_params.append(model_info["param"])

        all_params = sorted(set(all_params))
        ax.set_xticks(all_params)
        ax.set_xticklabels([f"{int(x)}" for x in all_params], weight="bold")

        ax.set_xlim(6, 750)

        current_values = self.results[mode][metric]
        min_val = min(current_values)
        max_val = max(current_values)

        if mode == "no_context" and metric == "accuracy":
            y_min = 0.45
            y_max = 0.60
            ax.set_ylim(y_min, y_max)

            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        else:
            if mode == "no_context":
                non_zero_values = [x for x in current_values if x != 0]
                min_val = min(non_zero_values)
                max_val = max(non_zero_values)

            y_min = math.floor(min_val * 20) / 20
            y_max = math.ceil(max_val * 20) / 20 + 0.01
            ax.set_ylim(y_min, y_max)

        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        ax.tick_params(axis="both")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight("bold")

    def draw_final(self):

        regular_font_path = "/Users/x/Library/Fonts/Lato-Regular.ttf"
        bold_font_path = "/Users/x/Library/Fonts/Lato-Bold.ttf"
        regular_font = fm.FontProperties(fname=regular_font_path)
        bold_font = fm.FontProperties(fname=bold_font_path)

        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(3, 5, height_ratios=[5, 5, 0.7])

        top_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
        bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]

        self.draw_single(
            "no_context",
            "precision",
            legend_on=False,
            ax=top_axes[0],
            show_y_label=True,
        )
        top_axes[0].set_xlabel(f"(a) Model Size (B)", fontproperties=bold_font)
        top_axes[0].set_ylabel("Precision", fontproperties=bold_font)
        top_axes[0].set_title(
            "w/o context, w/o revision", fontproperties=bold_font, fontsize=10
        )

        self.draw_single(
            "no_context",
            "recall",
            legend_on=False,
            ax=bottom_axes[0],
            show_y_label=True,
        )
        bottom_axes[0].set_xlabel(
            f"(b) Model Size (B)", fontproperties=bold_font, weight="bold"
        )
        bottom_axes[0].set_ylabel("Recall", fontproperties=bold_font, weight="bold")
        bottom_axes[0].set_title(
            "w/o context, w/o revision",
            fontproperties=bold_font,
            fontsize=10,
            weight="bold",
        )

        for i, metric in enumerate(["precision", "recall"]):
            if "o3-mini-medium" in self.model_meta_data:
                o3_idx = list(self.model_meta_data.keys()).index("o3-mini-medium")
                o3_value = self.results["no_context"][metric][o3_idx]
                x_lim = [top_axes, bottom_axes][i][0].get_xlim()
                [top_axes, bottom_axes][i][0].hlines(
                    y=o3_value,
                    xmin=x_lim[0],
                    xmax=x_lim[1],
                    colors=GPT_R_COLOR,
                    linestyles="--",
                    linewidth=2,
                )

        modes = ["no_context", "context", "normal", "hard"]
        for i, mode in enumerate(modes):
            self.draw_single(
                mode, "f1", legend_on=False, ax=top_axes[i + 1], show_y_label=(i == 0)
            )
            top_axes[i + 1].set_xlabel(
                f"({chr(99+i*2)}) Model Size (B)", fontproperties=bold_font
            )
            top_axes[i + 1].set_ylabel("F1", fontproperties=bold_font)
            if mode == "no_context":
                title = "w/o context, w/o revision"
            elif mode == "context":
                title = "w/ context, w/o revision"
            else:
                if mode == "normal":
                    title = f"Lenient Mode"
                elif mode == "hard":
                    title = f"Strict Mode"
            top_axes[i + 1].set_title(title, fontproperties=bold_font, fontsize=10)

        metrics_modes = [
            ("no_context", "accuracy"),
            ("hard", "precision"),
            ("hard", "recall"),
            ("hard", "accuracy"),
        ]

        for i, (mode, metric) in enumerate(metrics_modes):
            self.draw_single(
                mode,
                metric,
                legend_on=False,
                ax=bottom_axes[i + 1],
                show_y_label=(i == 0),
            )

            bottom_axes[i + 1].set_xlabel(
                f"({chr(100+i*2)}) Model Size (B)",
                fontproperties=bold_font,
                weight="bold",
            )

            bottom_axes[i + 1].set_ylabel(
                f"{metric.capitalize()}", fontproperties=bold_font, weight="bold"
            )

            if mode == "no_context":
                title = "w/o context, w/o revision"
            elif mode == "context":
                title = "w/ context, w/o revision"
            else:
                if mode == "normal":
                    title = f"Lenient Mode"
                elif mode == "hard":
                    title = f"Strict Mode"
            bottom_axes[i + 1].set_title(
                title, fontproperties=bold_font, fontsize=10, weight="bold"
            )

            for label in (
                bottom_axes[i + 1].get_xticklabels()
                + bottom_axes[i + 1].get_yticklabels()
            ):
                label.set_fontproperties(regular_font)

                if label.get_position()[1] == 0:
                    label.set_fontsize(label.get_fontsize() - 0.5)

            if True:

                o3_idx = list(self.model_meta_data.keys()).index("o3-mini-medium")
                o3_value = self.results[mode][metric][o3_idx]
                x_lim = bottom_axes[i + 1].get_xlim()
                bottom_axes[i + 1].hlines(
                    y=o3_value,
                    xmin=x_lim[0],
                    xmax=x_lim[1],
                    colors=GPT_R_COLOR,
                    linestyles="--",
                    linewidth=2,
                )

        for ax_row in [top_axes, bottom_axes]:
            for ax in ax_row:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(regular_font)

                    if label.get_position()[1] == 0:
                        label.set_fontsize(label.get_fontsize() - 0.5)

        legend_ax = fig.add_subplot(gs[2, :])
        legend_ax.axis("off")

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=LLAMA_R_COLOR,
                linestyle="-",
                marker="o",
                label="Llama (reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=LLAMA_NR_COLOR,
                linestyle="--",
                marker="o",
                label="Llama (non-reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_R_COLOR,
                linestyle="-",
                marker="s",
                label="Qwen (reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_NR_COLOR,
                linestyle="--",
                marker="s",
                label="Qwen (non-reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_R_COLOR,
                marker="*",
                label="DeepSeek-R1",
                linestyle="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_NR_COLOR,
                marker="*",
                label="DeepSeek-V3",
                linestyle="none",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                color=GPT_R_COLOR,
                linestyle="--",
                label="o3-mini",
                markersize=6,
            ),
        ]

        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            ncol=7,
            fontsize="x-small",
            columnspacing=1.0,
            handletextpad=0.5,
            bbox_to_anchor=(0.48, 1.1),
            frameon=True,
        )

        for text in legend.get_texts():
            text.set_fontproperties(regular_font)

        plt.tight_layout(h_pad=1, w_pad=1)
        plt.savefig("metrics_plots/rq1-f1.pdf", dpi=300, bbox_inches="tight")
        plt.close()
