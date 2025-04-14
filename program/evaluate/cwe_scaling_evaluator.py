from evaluate.base_evaluator import Evaluator
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from collections import defaultdict
import matplotlib.font_manager as fm
from common.cwe_analyzer import cwe_graph
from matplotlib.lines import Line2D
from scipy import stats
from common.globals import Args


DEEPSEEK_R_COLOR = "#5BB5AC"
DEEPSEEK_NR_COLOR = "#8bdad2"
QWEN_R_COLOR = "#D8B365"
QWEN_NR_COLOR = "#ebcb88"
LLAMA_R_COLOR = "#DE526C"
LLAMA_NR_COLOR = "#f37e94"
GPT_R_COLOR = "#82b0d2"


def parse_cwe(raw_cwe_id):
    raw_cwe_id = (
        raw_cwe_id.replace("'", "")
        .replace('"', "")
        .replace(" ", "")
        .replace("CWE-", "")
    )
    cwe_list = raw_cwe_id.split(",")
    return [int(cwe) for cwe in cwe_list if cwe]


def get_top_cwe_list(cwe_list):
    ret = set()
    for cwe in cwe_list:
        ret.add(cwe_graph.top_cwe(cwe))
    return list(ret)


class CWEScalingEvaluator(Evaluator):

    def __init__(self):
        super().__init__()
        self.drop_data("r1-7b")

    def calculate_f1_scores(self, cwe):

        param_f1_scores = defaultdict(list)

        for model_name, model_data in self.model_meta_data.items():

            if model_data["param"] > 100 and model_name not in ["v3", "r1"]:
                continue

            tp = fp = fn = tn = 0
            for result in model_data["data"].values():
                cwe_list = parse_cwe(result.meta_data["cwe"])
                top_cwe_list = get_top_cwe_list(cwe_list)

                if cwe not in top_cwe_list:
                    continue

                if result.hard_vuln_ret == 1:
                    tp += 1
                else:
                    fn += 1
                if result.hard_patched_ret == 0:
                    tn += 1
                else:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            param_f1_scores[model_data["param"]].append(
                {
                    "model": model_name,
                    "f1": f1,
                    "is_reasoning": model_data["is_reasoning"],
                }
            )

        return param_f1_scores

    def calculate_one_zero_rates(self, cwe):
        param_one_zero_rates = defaultdict(list)
        for model_name, model_data in self.model_meta_data.items():

            if model_data["param"] > 100 and model_name not in ["v3", "r1"]:
                continue

            total_count = 0
            one_zero_count = 0
            for result in model_data["data"].values():
                cwe_list = parse_cwe(result.meta_data["cwe"])
                top_cwe_list = get_top_cwe_list(cwe_list)

                if cwe not in top_cwe_list:
                    continue

                total_count += 1
                if result.hard_vuln_ret == 1 and result.hard_patched_ret == 0:
                    one_zero_count += 1

            one_zero_rate = one_zero_count / total_count if total_count > 0 else 0

            param_one_zero_rates[model_data["param"]].append(
                {
                    "model": model_name,
                    "rate": one_zero_rate,
                    "is_reasoning": model_data["is_reasoning"],
                }
            )

        return param_one_zero_rates

    def _set_axis_properties(self, ax, cwe):

        ax.set_xlim(6, 750)
        ax.set_xscale("log")

        all_params = []
        for model_name, model_info in self.model_meta_data.items():
            if model_info["param"] is not None and model_name != "qwq":
                all_params.append(model_info["param"])

        all_params = sorted(set(all_params))
        ax.set_xticks(all_params)
        ax.set_xticklabels([f"{int(x)}" for x in all_params])

        y_values = []
        for line in ax.get_lines():
            y_values.extend(line.get_ydata())

        if y_values:
            min_val = min(y_values)
            max_val = max(y_values)

            if cwe == 435:
                y_min = math.floor(min_val * 10) / 10
                y_max = math.ceil(max_val * 10) / 10 + 0.01
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
            else:
                y_min = math.floor(min_val * 20) / 20
                y_max = math.ceil(max_val * 20) / 20 + 0.01
                ax.set_ylim(y_min, y_max)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            if cwe != 435:
                ax.yaxis.get_major_locator().set_params(steps=[0.05, 0.05, 0.05])

    def plot_scaling_trends(self, cwe_sets=None, metrics=None):
        if cwe_sets is None:
            cwe_sets = {
                "all": set([435, 693, 284, 664, 682, 697, 703, 707, 691, 710]),
            }

        if metrics is None:
            metrics = ["f1", "precision", "recall"]

        regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )
        bold_font = fm.FontProperties(fname="/Users/x/Library/Fonts/Lato-Bold.ttf")

        for set_name, cwes in cwe_sets.items():
            for metric in metrics:

                fig = plt.figure(figsize=(12, 5))

                gs = fig.add_gridspec(3, 5, height_ratios=[5, 5, 0.7])

                top_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
                bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(5)]

                if set_name == "all":

                    cwes_list = [435, 693, 284, 664, 682, 697, 703, 707, 691, 710]
                else:

                    cwes_list = sorted(list(cwes))[:10]

                if len(cwes_list) < 10:
                    print(f"Warning: Only {len(cwes_list)} CWE(s) found")

                for idx, cwe in enumerate(cwes_list, 1):

                    ax = top_axes[idx - 1] if idx <= 5 else bottom_axes[idx - 6]

                    param_metrics = self.calculate_metrics(cwe)
                    y_label = metric.capitalize()

                    self._set_axis_properties(ax, cwe)

                    if cwe == 435:
                        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

                    if "o3-mini-medium" in self.model_meta_data:

                        tp = fp = fn = tn = 0
                        o3_data = self.model_meta_data["o3-mini-medium"]["data"]

                        for result in o3_data.values():
                            cwe_list = parse_cwe(result.meta_data["cwe"])
                            top_cwe_list = get_top_cwe_list(cwe_list)

                            if cwe not in top_cwe_list:
                                continue

                            if result.hard_vuln_ret == 1:
                                tp += 1
                            else:
                                fn += 1

                            if result.hard_patched_ret == 0:
                                tn += 1
                            else:
                                fp += 1

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = (
                            2 * (precision * recall) / (precision + recall)
                            if (precision + recall) > 0
                            else 0
                        )

                        o3_value = {"f1": f1, "precision": precision, "recall": recall}[
                            metric
                        ]

                        x_lim = ax.get_xlim()
                        ax.hlines(
                            y=o3_value,
                            xmin=x_lim[0],
                            xmax=x_lim[1],
                            colors=GPT_R_COLOR,
                            linestyles="--",
                            linewidth=2,
                        )

                    shadow_params = {
                        682: {
                            "multiplier": 1.5,
                            "non_reasoning_offset_minus": 0.075,
                            "non_reasoning_offset_add": 0.06,
                            "reasoning_offset_minus": 0.025,
                            "reasoning_offset_add": 0.028,
                        },
                        691: {
                            "multiplier": 1.5,
                            "non_reasoning_offset_minus": 0.028,
                            "non_reasoning_offset_add": 0.043,
                            "reasoning_offset_minus": 0.082,
                            "reasoning_offset_add": 0.073,
                        },
                        710: {
                            "multiplier": 1.5,
                            "non_reasoning_offset_minus": 0.047,
                            "non_reasoning_offset_add": 0.11,
                            "reasoning_offset_minus": 0.05,
                            "reasoning_offset_add": 0.04,
                        },
                        664: {
                            "multiplier": 1.5,
                            "non_reasoning_offset_minus": 0.05,
                            "non_reasoning_offset_add": 0.11,
                            "reasoning_offset_minus": 0.015,
                            "reasoning_offset_add": 0.015,
                        },
                    }

                    if cwe in shadow_params:

                        all_params = []
                        for model_info in self.model_meta_data.values():
                            if model_info["param"] is not None:
                                all_params.append(model_info["param"])
                        x_min, x_max = min(all_params), max(all_params)

                        reasoning_data = {"x": [], "y": []}
                        non_reasoning_data = {"x": [], "y": []}

                        for param, metrics_list in param_metrics.items():
                            for metrics_data in metrics_list:
                                if metrics_data["is_reasoning"]:
                                    reasoning_data["x"].append(param)
                                    reasoning_data["y"].append(metrics_data[metric])
                                else:
                                    non_reasoning_data["x"].append(param)
                                    non_reasoning_data["y"].append(metrics_data[metric])

                        params = shadow_params[cwe]

                        for data, color, alpha in [
                            (reasoning_data, "blue", 0.06),
                            (non_reasoning_data, "purple", 0.08),
                        ]:
                            if data["x"]:

                                slope, intercept, _, _, _ = stats.linregress(
                                    np.log10(data["x"]), data["y"]
                                )

                                x_smooth = np.geomspace(x_min - 6, x_max + 100, 100)
                                y_trend = slope * np.log10(x_smooth) + intercept

                                if color == "purple":
                                    offset_minus = params["non_reasoning_offset_minus"]
                                    offset_add = params["non_reasoning_offset_add"]
                                else:
                                    offset_minus = params["reasoning_offset_minus"]
                                    offset_add = params["reasoning_offset_add"]

                                ax.fill_between(
                                    x_smooth,
                                    y_trend - offset_minus * params["multiplier"],
                                    y_trend + offset_add * params["multiplier"],
                                    alpha=alpha,
                                    color=color,
                                )

                    llama_reasoning = {"params": [], "scores": []}
                    llama_non_reasoning = {"params": [], "scores": []}
                    qwen_reasoning = {"params": [], "scores": []}
                    qwen_non_reasoning = {"params": [], "scores": []}
                    qwq_data = {"params": [], "scores": []}
                    special_models = {}

                    for param, metrics_list in param_metrics.items():
                        for metrics_data in metrics_list:
                            model = metrics_data["model"]
                            score = metrics_data[metric]

                            if model in ["r1", "v3"]:
                                special_models[model] = {"param": param, "score": score}
                            elif model == "qwq":
                                qwq_data["params"].append(param)
                                qwq_data["scores"].append(score)
                            elif model in ["r1-8b", "r1-70b"]:
                                llama_reasoning["params"].append(param)
                                llama_reasoning["scores"].append(score)
                            elif model in ["8b", "70b"]:
                                llama_non_reasoning["params"].append(param)
                                llama_non_reasoning["scores"].append(score)
                            elif model in ["r1-7b", "r1-14b", "r1-32b"]:
                                qwen_reasoning["params"].append(param)
                                qwen_reasoning["scores"].append(score)
                            elif model in ["7b", "14b", "32b"]:
                                qwen_non_reasoning["params"].append(param)
                                qwen_non_reasoning["scores"].append(score)

                    def plot_series(data, color, style, marker, label):
                        if data["params"]:
                            sorted_idx = sorted(
                                range(len(data["params"])),
                                key=lambda k: data["params"][k],
                            )
                            x = [data["params"][i] for i in sorted_idx]
                            y = [data["scores"][i] for i in sorted_idx]
                            ax.plot(
                                x,
                                y,
                                style,
                                color=color,
                                linewidth=2,
                                marker=marker,
                                markersize=6,
                                label=label,
                            )

                    plot_series(
                        llama_reasoning, LLAMA_R_COLOR, "-", "o", "Llama (reasoning)"
                    )
                    plot_series(
                        llama_non_reasoning,
                        LLAMA_NR_COLOR,
                        "--",
                        "o",
                        "Llama (non-reasoning)",
                    )
                    plot_series(
                        qwen_reasoning, QWEN_R_COLOR, "-", "s", "Qwen (reasoning)"
                    )
                    plot_series(
                        qwen_non_reasoning,
                        QWEN_NR_COLOR,
                        "--",
                        "s",
                        "Qwen (non-reasoning)",
                    )

                    for model, data in special_models.items():
                        color = DEEPSEEK_R_COLOR if model == "r1" else DEEPSEEK_NR_COLOR
                        label = f"DeepSeek-{model}"
                        ax.plot(
                            [data["param"]],
                            [data["score"]],
                            "*",
                            color=color,
                            markersize=10,
                            label=label,
                        )

                    letter_idx = (
                        chr(97 + (idx - 1)) if idx <= 5 else chr(97 + 5 + (idx - 6))
                    )
                    ax.set_title(
                        f"CWE-{cwe}", fontproperties=bold_font, pad=5, fontsize=10
                    )
                    ax.set_xlabel(
                        f"({letter_idx}) Model Size (B)", fontproperties=bold_font
                    )

                    if True:
                        ax.set_ylabel(y_label, fontproperties=bold_font)

                    ax.grid(True, linestyle="--", alpha=0.2)

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
                    bbox_to_anchor=(0.48, 1.4),
                    frameon=True,
                )

                for text in legend.get_texts():
                    text.set_fontproperties(regular_font)

                for ax_row in [top_axes, bottom_axes]:
                    for ax in ax_row:

                        y_values = []
                        for line in ax.get_lines():
                            y_values.extend(line.get_ydata())

                        if y_values:
                            min_val = min(y_values)
                            max_val = max(y_values)

                            title = ax.get_title()
                            current_cwe = int(title.replace("CWE-", ""))

                            if current_cwe == 435:
                                y_min = math.floor(min_val * 10) / 10
                                y_max = math.ceil(max_val * 10) / 10
                                if y_max - y_min < 0.1:
                                    y_max = y_min + 0.1
                                ax.set_ylim(y_min, y_max)
                                ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
                            else:
                                y_min = math.floor(min_val * 20) / 20
                                y_max = math.ceil(max_val * 20) / 20
                                if y_max - y_min < 0.05:
                                    y_max = y_min + 0.05
                                ax.set_ylim(y_min, y_max)
                                ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

                            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

                plt.tight_layout(h_pad=1, w_pad=1)

                os.makedirs("metrics_plots", exist_ok=True)
                plt.savefig(
                    f"metrics_plots/cwe_scaling_{set_name}_{metric}.pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()

    def calculate_metrics(self, cwe):

        param_metrics = defaultdict(list)

        for model_name, model_data in self.model_meta_data.items():

            if model_data["param"] > 100 and model_name not in ["v3", "r1"]:
                continue

            tp = fp = fn = tn = 0
            for result in model_data["data"].values():
                cwe_list = parse_cwe(result.meta_data["cwe"])
                top_cwe_list = get_top_cwe_list(cwe_list)

                if cwe not in top_cwe_list:
                    continue

                if result.hard_vuln_ret == 1:
                    tp += 1
                else:
                    fn += 1

                if result.hard_patched_ret == 0:
                    tn += 1
                else:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            param_metrics[model_data["param"]].append(
                {
                    "model": model_name,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "is_reasoning": model_data["is_reasoning"],
                }
            )

        return param_metrics

    def evaluate(self, cwe_sets=None):
        if cwe_sets is None:
            cwe_sets = {
                "all": set([435, 693, 284, 664, 682, 697, 703, 707, 691, 710]),
            }

        self.plot_scaling_trends(cwe_sets, metrics=["f1", "precision", "recall"])

        print("\nCWE F1-score：")
        for set_name, cwes in cwe_sets.items():
            print(f"\nAnalyzing{set_name}：")

            cwe_metrics = {metric: {} for metric in ["f1", "precision", "recall"]}
            max_metrics = {metric: {} for metric in ["f1", "precision", "recall"]}
            if set_name == "all":
                cwe_order = [435, 693, 284, 664, 682, 697, 703, 707, 691, 710]

                cwe_list = [cwe for cwe in cwe_order if cwe in cwes]
            else:
                cwe_list = sorted(list(cwes))

            for cwe in cwe_list:
                metrics = self.calculate_metrics(cwe)

                for metric_name in ["f1", "precision", "recall"]:
                    all_values = []
                    model_value_map = {}

                    for param_metrics in metrics.values():
                        for metric_data in param_metrics:
                            model_value_map[metric_data["model"]] = metric_data[
                                metric_name
                            ]
                            all_values.append(metric_data[metric_name])

                    avg_value = sum(all_values) / len(all_values) if all_values else 0
                    cwe_metrics[metric_name][cwe] = avg_value
                    max_metrics[metric_name][cwe] = max(all_values)
                    improvements = {"qwen": [], "llama": [], "all": []}

                    if all(m in model_value_map for m in ["r1-14b", "14b"]):
                        diff_14b = model_value_map["r1-14b"] - model_value_map["14b"]
                        improvements["qwen"].append(diff_14b)
                        improvements["all"].append(diff_14b)

                    if all(m in model_value_map for m in ["r1-32b", "32b"]):
                        diff_32b = model_value_map["r1-32b"] - model_value_map["32b"]
                        improvements["qwen"].append(diff_32b)
                        improvements["all"].append(diff_32b)

                    if all(m in model_value_map for m in ["r1-8b", "8b"]):
                        diff_8b = model_value_map["r1-8b"] - model_value_map["8b"]
                        improvements["llama"].append(diff_8b)
                        improvements["all"].append(diff_8b)

                    if all(m in model_value_map for m in ["r1-70b", "70b"]):
                        diff_70b = model_value_map["r1-70b"] - model_value_map["70b"]
                        improvements["llama"].append(diff_70b)
                        improvements["all"].append(diff_70b)

                    if all(m in model_value_map for m in ["r1", "v3"]):
                        diff_deepseek = model_value_map["r1"] - model_value_map["v3"]
                        improvements["all"].append(diff_deepseek)

                    all_avg = (
                        sum(improvements["all"]) / len(improvements["all"])
                        if improvements["all"]
                        else 0
                    )

                print(f"\nCWE-{cwe}:")
                for metric_name in ["f1", "precision", "recall"]:
                    print(
                        f"Avg {metric_name.capitalize()}: {cwe_metrics[metric_name][cwe]:.3f}"
                    )
                    print(
                        f"Max {metric_name.capitalize()}: {max_metrics[metric_name][cwe]:.3f}"
                    )

            print(f"\nAvg metrics of all CWE:")
            for metric_name in ["f1", "precision", "recall"]:
                avg_metric = sum(cwe_metrics[metric_name].values()) / len(
                    cwe_metrics[metric_name]
                )
                max_metric = max(cwe_metrics[metric_name].values())
                print(f"Avg {metric_name.capitalize()}: {avg_metric:.3f}")
                print(f"Max {metric_name.capitalize()}: {max_metric:.3f}")

        print("\n\nDetailed metrics of each CWE:")
        print("CWE\tF1\tPrecision\tRecall")
        print("-" * 40)

        all_cwes = set()
        for cwes_set in cwe_sets.values():
            all_cwes.update(cwes_set)

        cwe_order = [435, 693, 284, 664, 682, 697, 703, 707, 691, 710]

        cwe_list = [cwe for cwe in cwe_order if cwe in all_cwes] + [
            cwe for cwe in sorted(all_cwes) if cwe not in cwe_order
        ]

        cwe_all_metrics = {}
        for cwe in cwe_list:
            metrics_data = self.calculate_metrics(cwe)

            f1_values = []
            precision_values = []
            recall_values = []

            for param_metrics in metrics_data.values():
                for metric_data in param_metrics:
                    f1_values.append(metric_data["f1"])
                    precision_values.append(metric_data["precision"])
                    recall_values.append(metric_data["recall"])

            avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
            avg_precision = (
                sum(precision_values) / len(precision_values) if precision_values else 0
            )
            avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0

            cwe_all_metrics[cwe] = {
                "f1": avg_f1,
                "precision": avg_precision,
                "recall": avg_recall,
            }

            print(f"{cwe}\t{avg_f1:.3f}\t{avg_precision:.3f}\t{avg_recall:.3f}")

        avg_f1 = sum(m["f1"] for m in cwe_all_metrics.values()) / len(cwe_all_metrics)
        avg_precision = sum(m["precision"] for m in cwe_all_metrics.values()) / len(
            cwe_all_metrics
        )
        avg_recall = sum(m["recall"] for m in cwe_all_metrics.values()) / len(
            cwe_all_metrics
        )

        print("-" * 40)
        print(f"Avg\t{avg_f1:.3f}\t{avg_precision:.3f}\t{avg_recall:.3f}")
