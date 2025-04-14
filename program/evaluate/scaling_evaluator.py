from evaluate.base_evaluator import Evaluator
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List, Any
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats
import tiktoken


def parse_result(final_response):
    if (
        "HAS_VUL" in final_response
        or "has_vul" in final_response
        or "YES_VUL" in final_response
        or "HAS\_VUL" in final_response
        or "HAS\\_VUL" in final_response
    ):
        return 1
    elif (
        "NO_VUL" in final_response
        or "no_vul" in final_response
        or "NO\\_VUL" in final_response
        or "NO\_VUL" in final_response
    ):
        return 0
    return -1


def parse_result_list(result_list):
    ret_list = []
    for result in result_list:
        entry = result
        entry["result"] = parse_result(result["final_response"])
        ret_list.append(entry)
    return ret_list


def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy


def analyze_results(
    data: Dict[str, List[Dict[str, List]]], max_tokens: int = 1000
) -> Dict[str, Any]:

    stats = defaultdict(int)

    tp = fp = tn = fn = 0

    first_tokens = 0

    for project, results in data.items():
        result = results[0]
        vuln_ret_list = result["vuln"]
        patched_ret_list = result["patched"]
        vuln_ret_list = parse_result_list(vuln_ret_list)
        patched_ret_list = parse_result_list(patched_ret_list)
        max_vuln = vuln_ret_list[0]
        first_tokens += vuln_ret_list[0]["thinking_tokens"]
        for i, result in enumerate(vuln_ret_list):
            if result["thinking_tokens"] < max_tokens:
                max_vuln = vuln_ret_list[i]
            else:
                break
        max_patched = patched_ret_list[0]
        first_tokens += patched_ret_list[0]["thinking_tokens"]
        for i, result in enumerate(patched_ret_list):
            if result["thinking_tokens"] < max_tokens:
                max_patched = patched_ret_list[i]
            else:
                break
        vuln_result = max_vuln["result"]
        if "ref" in max_vuln:
            ref_entry = vuln_ret_list[max_vuln["ref"]]
            if ref_entry["eval"]["ret"]:
                vuln_result = 1
            else:
                vuln_result = 0
        patched_result = max_patched["result"]
        if "ref" in max_patched:
            ref_entry = patched_ret_list[max_patched["ref"]]
            if ref_entry["eval"]["ret"]:
                patched_result = 1
            else:
                patched_result = 0
        if vuln_result == 1:
            tp += 1
        elif vuln_result == 0:
            fn += 1
        if patched_result == 1:
            fp += 1
        elif patched_result == 0:
            tn += 1
    avg_first_tokens = first_tokens / (2 * len(data))
    precision, recall, f1, accuracy = calculate_metrics(tp, fp, tn, fn)
    stats["precision"] = precision
    stats["recall"] = recall
    stats["f1"] = f1
    stats["avg_tokens"] = avg_first_tokens
    stats["accuracy"] = accuracy
    stats["tp"] = tp
    stats["tn"] = tn
    stats["fp"] = fp
    stats["fn"] = fn
    return dict(stats)


def get_avg_thinking_tokens(res, key, k_list, is_vuln):
    total_tokens = 0
    if is_vuln:
        for k in k_list:
            total_tokens += res[key][k]["vuln"]["thinking_tokens"]
    else:
        for k in k_list:
            total_tokens += res[key][k]["patched"]["thinking_tokens"]
    return total_tokens


def get_single_result(res, key, k, is_vuln):
    try:
        if is_vuln:
            if res[key][k]["vuln"]["cot"]["ret"] == 1:
                if res[key][k]["vuln"]["eval"]["ret"]:
                    return 1, 0, 0, 0
                else:
                    return 0, 0, 1, 0
            elif res[key][k]["vuln"]["cot"]["ret"] == 0:
                return 0, 0, 1, 0
        else:
            if res[key][k]["patched"]["cot"]["ret"] == 1:
                if res[key][k]["patched"]["eval"]["ret"]:
                    return 0, 1, 0, 0
                else:
                    return 0, 0, 0, 1
            elif res[key][k]["patched"]["cot"]["ret"] == 0:
                return 0, 0, 0, 1
        return 0, 0, 0, 0
    except:
        print(f"Error in get_single_result: {key}, {k}, {is_vuln}")
        return 0, 0, 0, 0


def analyze_all_data(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, precision, recall, f1


def analyze_par_result(majority_file: str, res_file: str):
    with open(majority_file, "r") as f:
        majority_res = json.load(f)
    with open(res_file, "r") as f:
        res = json.load(f)

    data = {
        "1": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "all_thinking_tokens": 0,
            "entry_num": 0,
            "avg_thinking_tokens": 0,
        },
        "3": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "all_thinking_tokens": 0,
            "entry_num": 0,
            "avg_thinking_tokens": 0,
        },
        "5": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "all_thinking_tokens": 0,
            "entry_num": 0,
            "avg_thinking_tokens": 0,
        },
        "8": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "acc": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "all_thinking_tokens": 0,
            "entry_num": 0,
            "avg_thinking_tokens": 0,
        },
    }

    for key, response_list in majority_res["vuln"].items():
        k1 = 0
        k3 = response_list["3"]["selected"]
        k5 = response_list["5"]["selected"]
        k8 = response_list["8"]["selected"]

        tp, fp, fn, tn = get_single_result(res, key, k1, True)
        data["1"]["tp"] += tp
        data["1"]["fp"] += fp
        data["1"]["fn"] += fn
        data["1"]["tn"] += tn
        data["1"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [k1], True
        )
        data["1"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k3, True)
        data["3"]["tp"] += tp
        data["3"]["fp"] += fp
        data["3"]["fn"] += fn
        data["3"]["tn"] += tn
        data["3"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3], True
        )
        data["3"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k5, True)
        data["5"]["tp"] += tp
        data["5"]["fp"] += fp
        data["5"]["fn"] += fn
        data["5"]["tn"] += tn
        data["5"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3, 4, 5], True
        )
        data["5"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k8, True)
        data["8"]["tp"] += tp
        data["8"]["fp"] += fp
        data["8"]["fn"] += fn
        data["8"]["tn"] += tn
        data["8"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3, 4, 5, 6, 7, 8], True
        )
        data["8"]["entry_num"] += 1

    for key, response_list in majority_res["patched"].items():
        k1 = 0
        k3 = response_list["3"]["selected"]
        k5 = response_list["5"]["selected"]
        k8 = response_list["8"]["selected"]

        tp, fp, fn, tn = get_single_result(res, key, k1, False)
        data["1"]["tp"] += tp
        data["1"]["fp"] += fp
        data["1"]["fn"] += fn
        data["1"]["tn"] += tn
        data["1"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [k1], False
        )
        data["1"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k3, False)
        data["3"]["tp"] += tp
        data["3"]["fp"] += fp
        data["3"]["fn"] += fn
        data["3"]["tn"] += tn
        data["3"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3], False
        )
        data["3"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k5, False)
        data["5"]["tp"] += tp
        data["5"]["fp"] += fp
        data["5"]["fn"] += fn
        data["5"]["tn"] += tn
        data["5"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3, 4, 5], False
        )
        data["5"]["entry_num"] += 1
        tp, fp, fn, tn = get_single_result(res, key, k8, False)
        data["8"]["tp"] += tp
        data["8"]["fp"] += fp
        data["8"]["fn"] += fn
        data["8"]["tn"] += tn
        data["8"]["all_thinking_tokens"] += get_avg_thinking_tokens(
            res, key, [1, 2, 3, 4, 5, 6, 7, 8], False
        )
        data["8"]["entry_num"] += 1

    for key, value in data.items():
        try:

            acc, precision, recall, f1 = analyze_all_data(
                value["tp"], value["fp"], value["fn"], value["tn"]
            )
            value["acc"] = acc
            value["precision"] = precision
            value["recall"] = recall
            value["f1"] = f1
            value["avg_thinking_tokens"] = (
                value["all_thinking_tokens"] / value["entry_num"]
            )
        except Exception as e:
            print(f"Error for {key}: {e}")

    return data


cwe_664_par_res = analyze_par_result(
    "../results/par_scaling/selected_data_664.json",
    "../results/par_scaling/result_664_thinking.json",
)
cwe_691_par_res = analyze_par_result(
    "../results/par_scaling/selected_data_691.json",
    "../results/par_scaling/result_691_thinking.json",
)
cwe_707_par_res = analyze_par_result(
    "../results/par_scaling/selected_data_707.json",
    "../results/par_scaling/result_707_thinking.json",
)
cwe_aggr_par_res = analyze_par_result(
    "../results/par_scaling/selected_data_final.json",
    "../results/par_scaling/result_final_thinking.json",
)


DEEPSEEK_R_COLOR = "#5BB5AC"
DEEPSEEK_NR_COLOR = "#8bdad2"
QWEN_R_COLOR = "#D8B365"
QWEN_NR_COLOR = "#ebcb88"
LLAMA_R_COLOR = "#DE526C"
LLAMA_NR_COLOR = "#f37e94"
GPT_R_COLOR = "#82b0d2"
GPT_NR_COLOR = "#c1d8eb"
HIGH_COLOR = "#246593"


class ScalingEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.drop_data("o3-mini-medium")

        self.load_data("o3-mini-high", 671, True, "o3-mini-high")
        self.load_data("o3-mini-low", 671, True, "o3-mini-low")
        self.load_data("o3-mini-medium", 671, True, "o3-mini-medium")
        self.scaling_data = {}
        self.load_scaling_data()

        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def load_scaling_data(self):
        data_path = "../results/seq_scaling/result_{}.json"
        for cwe in [691, 707, 664]:
            with open(data_path.format(cwe), "r", encoding="utf-8") as f:
                data = json.load(f)
                self.scaling_data[cwe] = data

    def remove_dup(self, str):
        if not str:
            return str

        words = str.split()
        if len(words) < 2:
            return str

        n = len(words)
        max_repeat_len = 0
        for i in range(1, n // 2 + 1):

            pattern = words[-i:]

            repeat_count = 0
            j = n - i
            while j >= i and words[j - i : j] == pattern:
                repeat_count += 1
                j -= i

            if repeat_count > 0:
                current_repeat_len = i * repeat_count
                if current_repeat_len > max_repeat_len:
                    max_repeat_len = current_repeat_len
                    max_pattern_len = i

        if max_repeat_len > 0:
            return " ".join(words[:-max_repeat_len])

        return str

    def get_length(self, str):

        if not str:
            return 0

        return len(self.tokenizer.encode(str))

    def get_average_reasoning_length(self, data):

        results = {
            "no_context": {"total_length": 0, "count": 0},
            "context": {"total_length": 0, "count": 0},
            "normal": {"total_length": 0, "count": 0},
            "hard": {"total_length": 0, "count": 0},
        }

        max_length = 1000000

        for result in data.values():

            if (
                hasattr(result, "hard_vuln_reasoning_tokens")
                and hasattr(result, "hard_patched_reasoning_tokens")
                and result.hard_vuln_reasoning_tokens > 0
                and result.hard_patched_reasoning_tokens > 0
            ):

                results["normal"]["total_length"] += (
                    result.hard_vuln_reasoning_tokens
                    + result.hard_patched_reasoning_tokens
                )
                results["normal"]["count"] += 2
                results["hard"]["total_length"] += (
                    result.hard_vuln_reasoning_tokens
                    + result.hard_patched_reasoning_tokens
                )
                results["hard"]["count"] += 2
                continue

            if hasattr(result, "no_context_vuln_output") and hasattr(
                result, "no_context_vuln_ret"
            ):
                reasoning = result.no_context_vuln_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.no_context_vuln_ret != -1
                ):
                    results["no_context"]["total_length"] += self.get_length(reasoning)
                    results["no_context"]["count"] += 1
            if hasattr(result, "no_context_patched_output") and hasattr(
                result, "no_context_patched_ret"
            ):
                reasoning = result.no_context_patched_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.no_context_patched_ret != -1
                ):
                    results["no_context"]["total_length"] += self.get_length(reasoning)
                    results["no_context"]["count"] += 1

            if hasattr(result, "context_vuln_output") and hasattr(
                result, "context_vuln_ret"
            ):
                reasoning = result.context_vuln_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.context_vuln_ret != -1
                ):
                    results["context"]["total_length"] += self.get_length(reasoning)
                    results["context"]["count"] += 1

            if hasattr(result, "context_patched_output") and hasattr(
                result, "context_patched_ret"
            ):
                reasoning = result.context_patched_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.context_patched_ret != -1
                ):
                    results["context"]["total_length"] += self.get_length(reasoning)
                    results["context"]["count"] += 1

            if hasattr(result, "normal_vuln_output") and hasattr(
                result, "normal_vuln_ret"
            ):
                reasoning = result.normal_vuln_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.normal_vuln_ret != -1
                ):
                    results["normal"]["total_length"] += self.get_length(reasoning)
                    results["normal"]["count"] += 1

            if hasattr(result, "normal_patched_output") and hasattr(
                result, "normal_patched_ret"
            ):
                reasoning = result.normal_patched_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.normal_patched_ret != -1
                ):
                    results["normal"]["total_length"] += self.get_length(reasoning)
                    results["normal"]["count"] += 1

            if hasattr(result, "hard_vuln_output") and hasattr(result, "hard_vuln_ret"):
                reasoning = result.hard_vuln_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.hard_vuln_ret != -1
                ):
                    results["hard"]["total_length"] += self.get_length(reasoning)
                    results["hard"]["count"] += 1

            if hasattr(result, "hard_patched_output") and hasattr(
                result, "hard_patched_ret"
            ):
                reasoning = result.hard_patched_output
                if (
                    reasoning
                    and self.get_length(reasoning) <= max_length
                    and result.hard_patched_ret != -1
                ):
                    results["hard"]["total_length"] += self.get_length(reasoning)
                    results["hard"]["count"] += 1

            if hasattr(result, "hard_patched_feedback") and hasattr(
                result, "hard_patched_ret"
            ):
                feedback = result.hard_patched_feedback
                if feedback["k"] != 0 and feedback["ret"] != -1:
                    for item in feedback["history"]:
                        if (
                            "rationale" in item
                            and len(item["rationale"]) <= max_length
                            and result.hard_patched_ret != -1
                        ):
                            results["hard"]["total_length"] += len(item["rationale"])
                            results["hard"]["count"] += 1

        final_results = {}
        for eval_type in results:
            if results[eval_type]["count"] > 0:
                final_results[eval_type] = (
                    results[eval_type]["total_length"] / results[eval_type]["count"]
                )
            else:
                final_results[eval_type] = 0.0

        return final_results

    def calculate_metrics(self, data, eval_type):
        """计算指定评估类型的指标"""
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
            except AttributeError:
                continue

        total = tp + fp + tn + fn
        if total == 0:
            return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def draw_metrics_vs_tokens(self):

        regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )
        bold_font = fm.FontProperties(fname="/Users/x/Library/Fonts/Lato-Bold.ttf")

        model_data = {}
        for model_name, model_info in self.model_meta_data.items():
            data = model_info["data"]

            avg_lengths = self.get_average_reasoning_length(data)

            metrics_values = self.calculate_metrics(data, "normal")

            model_data[model_name] = {
                "tokens": avg_lengths["normal"],
                "metrics": metrics_values,
            }

        metrics = ["Accuracy", "Precision", "Recall"]

        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 0.7])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            llama_reasoning_data = {"tokens": [], "values": [], "names": []}
            qwen_reasoning_data = {"tokens": [], "values": [], "names": []}
            llama_non_reasoning_data = {"tokens": [], "values": [], "names": []}
            qwen_non_reasoning_data = {"tokens": [], "values": [], "names": []}
            deepseek_data = {"tokens": [], "values": [], "names": []}
            o3_mini_data = {"tokens": [], "values": [], "names": []}

            for model_name, data in model_data.items():
                if not data["tokens"]:
                    continue

                metric_key = metric.lower()
                metric_value = data["metrics"][metric_key]

                if model_name in ["r1-8b", "r1-70b"]:
                    llama_reasoning_data["tokens"].append(data["tokens"])
                    llama_reasoning_data["values"].append(metric_value)
                    llama_reasoning_data["names"].append(model_name)

                elif model_name in ["r1-7b", "r1-14b", "r1-32b"]:
                    qwen_reasoning_data["tokens"].append(data["tokens"])
                    qwen_reasoning_data["values"].append(metric_value)
                    qwen_reasoning_data["names"].append(model_name)

                elif model_name in ["r1", "v3"]:
                    deepseek_data["tokens"].append(data["tokens"])
                    deepseek_data["values"].append(metric_value)
                    deepseek_data["names"].append(model_name)

                elif model_name in ["8b", "70b"]:
                    llama_non_reasoning_data["tokens"].append(data["tokens"])
                    llama_non_reasoning_data["values"].append(metric_value)
                    llama_non_reasoning_data["names"].append(model_name)

                elif model_name in ["7b", "14b", "32b"]:
                    qwen_non_reasoning_data["tokens"].append(data["tokens"])
                    qwen_non_reasoning_data["values"].append(metric_value)
                    qwen_non_reasoning_data["names"].append(model_name)

                elif model_name in ["o3-mini-medium"]:
                    o3_mini_data["tokens"].append(data["tokens"])
                    o3_mini_data["values"].append(metric_value)
                    o3_mini_data["names"].append(model_name)

            for data_dict, color, marker, label in [
                (llama_reasoning_data, LLAMA_R_COLOR, "o", "Llama (w/ reasoning)"),
                (qwen_reasoning_data, QWEN_R_COLOR, "s", "Qwen (w/ reasoning)"),
                (
                    llama_non_reasoning_data,
                    LLAMA_NR_COLOR,
                    "o",
                    "Llama (w/o reasoning)",
                ),
                (qwen_non_reasoning_data, QWEN_NR_COLOR, "s", "Qwen (w/o reasoning)"),
                (deepseek_data, DEEPSEEK_R_COLOR, "*", "Deepseek"),
                (o3_mini_data, GPT_R_COLOR, "D", "o3-mini"),
            ]:
                if data_dict["tokens"]:

                    ax.scatter(
                        data_dict["tokens"],
                        data_dict["values"],
                        c=color,
                        s=80,
                        marker=marker,
                        label=label,
                    )

                    for i, (x, y, name) in enumerate(
                        zip(
                            data_dict["tokens"], data_dict["values"], data_dict["names"]
                        )
                    ):

                        if "Llama" in label:
                            y_offset = 0.01
                            x_offset = 0
                        elif "Qwen" in label:
                            y_offset = -0.01
                            x_offset = 0
                        elif "Deepseek" in label:
                            y_offset = 0.015
                            x_offset = 0
                        else:
                            y_offset = 0.01
                            x_offset = 0

                        ax.annotate(
                            name,
                            xy=(x, y),
                            xytext=(x_offset, y_offset),
                            textcoords="offset points",
                            fontproperties=regular_font,
                            fontsize=7,
                            ha="center",
                            va="bottom",
                        )

            ax.set_xlabel(
                f"({chr(97+idx)}) Average Thinking Tokens", fontproperties=bold_font
            )
            ax.set_ylabel(f"{metric}", fontproperties=bold_font)
            ax.set_title(f"{metric}", fontproperties=bold_font, fontsize=10)

            ax.grid(True, which="major", ls="-", alpha=0.2)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(regular_font)
                label.set_fontsize(8)

            ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

            ax.set_xlim(500, 1700)

        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=LLAMA_R_COLOR,
                marker="o",
                label="Llama (w/ reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=LLAMA_NR_COLOR,
                marker="o",
                label="Llama (w/o reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_R_COLOR,
                marker="s",
                label="Qwen (w/ reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_NR_COLOR,
                marker="s",
                label="Qwen (w/o reasoning)",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_R_COLOR,
                marker="*",
                label="Deepseek",
                markersize=10,
            ),
            Line2D(
                [0], [0], color=GPT_R_COLOR, marker="D", label="o3-mini", markersize=6
            ),
        ]

        legend = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            ncol=3,
            fontsize="x-small",
            columnspacing=1.0,
            handletextpad=0.5,
            bbox_to_anchor=(0.48, 1.1),
            frameon=True,
        )

        for text in legend.get_texts():
            text.set_fontproperties(regular_font)

        plt.tight_layout(h_pad=1.5, w_pad=2.0)
        plt.savefig("metrics_plots/metrics_vs_tokens.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def evaluate(self):

        cwe_data = defaultdict(lambda: defaultdict(list))
        metrics = ["Accuracy", "Precision", "Recall"]
        models = ["o3-mini-high", "o3-mini-low", "o3-mini-medium"]
        cwes = [691, 707, 664, "combined"]

        for model_name in models:
            data = self.model_meta_data[model_name]["data"]
            for item in data.values():
                top_cwe = item.meta_data["top-cwe"]

                avg_tokens = (
                    item.hard_vuln_reasoning_tokens + item.hard_patched_reasoning_tokens
                ) / 2

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                if item.normal_vuln_ret == 1:
                    tp += 1
                elif item.normal_vuln_ret == 0:
                    fn += 1

                if item.normal_patched_ret == 0:
                    tn += 1
                elif item.normal_patched_ret == 1:
                    fp += 1
                cwe_data[top_cwe][model_name].append(
                    {"tokens": avg_tokens, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
                )
                if top_cwe in [691, 707, 664]:

                    cwe_data["combined"][model_name].append(
                        {"tokens": avg_tokens, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
                    )

        fig, axes = plt.subplots(2, 6, figsize=(9.5, 3.6))

        regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )
        bold_font = fm.FontProperties(fname="/Users/x/Library/Fonts/Lato-Bold.ttf")

        all_model_metrics = {}
        for cwe in cwes:
            all_model_metrics[cwe] = defaultdict(dict)

            if cwe in self.scaling_data or cwe == "combined":
                scaling_metrics = []

                exponent = 1.1
                token_steps = [
                    800 * exponent**i for i in range(100) if 800 * exponent**i <= 10000
                ]

                if cwe == "combined":

                    for token_limit in token_steps:
                        combined_result = {
                            "tp": 0,
                            "tn": 0,
                            "fp": 0,
                            "fn": 0,
                            "tokens": token_limit,
                        }
                        for sub_cwe in [691, 707, 664]:
                            result = analyze_results(
                                self.scaling_data[sub_cwe], max_tokens=token_limit
                            )

                            for key in ["tp", "tn", "fp", "fn"]:
                                combined_result[key] += result.get(key, 0)

                        precision, recall, f1, accuracy = calculate_metrics(
                            combined_result["tp"],
                            combined_result["fp"],
                            combined_result["tn"],
                            combined_result["fn"],
                        )

                        scaling_metrics.append(
                            {
                                "tokens": token_limit,
                                "F1": f1,
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                            }
                        )
                else:

                    for token_limit in token_steps:
                        result = analyze_results(
                            self.scaling_data[cwe], max_tokens=token_limit
                        )
                        scaling_metrics.append(
                            {
                                "tokens": token_limit,
                                "F1": result["f1"],
                                "Accuracy": result["accuracy"],
                                "Precision": result["precision"],
                                "Recall": result["recall"],
                            }
                        )

                all_model_metrics[cwe]["scaling"] = scaling_metrics

            par_res = None
            if cwe == 664:
                par_res = cwe_664_par_res
            elif cwe == 691:
                par_res = cwe_691_par_res
            elif cwe == 707:
                par_res = cwe_707_par_res
            elif cwe == "combined":
                par_res = cwe_aggr_par_res

            if par_res:
                par_scaling_metrics = []
                for k in ["1", "3", "5", "8"]:
                    par_scaling_metrics.append(
                        {
                            "tokens": par_res[k]["avg_thinking_tokens"],
                            "F1": par_res[k]["f1"],
                            "Accuracy": par_res[k]["acc"],
                            "Precision": par_res[k]["precision"],
                            "Recall": par_res[k]["recall"],
                        }
                    )
                all_model_metrics[cwe]["par_scaling"] = par_scaling_metrics

            for model in models:
                data = cwe_data[cwe][model]
                if not data:
                    continue

                total_tp = sum(d["tp"] for d in data)
                total_tn = sum(d["tn"] for d in data)
                total_fp = sum(d["fp"] for d in data)
                total_fn = sum(d["fn"] for d in data)
                avg_tokens = sum(d["tokens"] for d in data) / len(data)

                precision = (
                    total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
                )
                recall = (
                    total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                accuracy = (
                    (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
                    if len(data) > 0
                    else 0
                )

                all_model_metrics[cwe][model] = {
                    "tokens": avg_tokens,
                    "F1": f1,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                }

        for metric_idx, metric in enumerate(metrics):
            for cwe_idx, cwe in enumerate(cwes):

                col = (metric_idx * 2) + (cwe_idx // 2)
                row = cwe_idx % 2

                ax = axes[row, col]
                model_metrics = all_model_metrics[cwe]

                x_values = [
                    model_metrics[model]["tokens"]
                    for model in models
                    if model in model_metrics
                ]
                y_values = [
                    model_metrics[model][metric]
                    for model in models
                    if model in model_metrics
                ]
                model_names = [model for model in models if model in model_metrics]

                for i, model in enumerate(model_names):
                    color = (
                        GPT_NR_COLOR
                        if "low" in model
                        else (GPT_R_COLOR if "medium" in model else HIGH_COLOR)
                    )
                    ax.scatter(x_values[i], y_values[i], c=color, s=30, marker="o")

                if len(x_values) > 1:
                    log_x_values = np.log10(x_values)
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        log_x_values, y_values
                    )

                    line_log_x = np.array([np.log10(1), np.log10(11000)])
                    line_y = slope * line_log_x + intercept
                    line_x = 10**line_log_x

                    ax.plot(
                        line_x,
                        line_y,
                        "--",
                        color=GPT_R_COLOR,
                        linewidth=1.5,
                        alpha=0.6,
                    )

                    if metric == "Accuracy":
                        offset = 0.01
                        x_smooth = np.geomspace(1, 11000, 100)
                        y_trend = slope * np.log10(x_smooth) + intercept
                        ax.fill_between(
                            x_smooth,
                            y_trend - offset * 1.3,
                            y_trend + offset * 1.3,
                            alpha=0.06,
                            color="blue",
                            label="_nolegend_",
                        )

                if "scaling" in model_metrics:
                    scaling_x = [m["tokens"] for m in model_metrics["scaling"]]
                    scaling_y = [m[metric] for m in model_metrics["scaling"]]

                    ax.scatter(
                        scaling_x,
                        scaling_y,
                        c=QWEN_R_COLOR,
                        s=15,
                        marker="s",
                        alpha=0.3,
                        label="r1-qn-14b Trend",
                    )

                    ax.plot(
                        scaling_x,
                        scaling_y,
                        "-",
                        color=QWEN_R_COLOR,
                        linewidth=1.2,
                        alpha=0.6,
                    )

                if "par_scaling" in model_metrics:
                    par_scaling_x = [m["tokens"] for m in model_metrics["par_scaling"]]
                    par_scaling_y = [m[metric] for m in model_metrics["par_scaling"]]

                    ax.scatter(
                        par_scaling_x,
                        par_scaling_y,
                        c=DEEPSEEK_R_COLOR,
                        s=20,
                        marker="^",
                        alpha=0.7,
                        label="r1-qn-14b(USC)",
                    )

                    ax.plot(
                        par_scaling_x,
                        par_scaling_y,
                        "-",
                        color=DEEPSEEK_R_COLOR,
                        linewidth=1.2,
                        alpha=0.7,
                    )

                if cwe in self.scaling_data:
                    result = analyze_results(self.scaling_data[cwe])
                    avg_tokens = result["avg_tokens"]
                    ax.axvline(
                        x=avg_tokens,
                        color=LLAMA_R_COLOR,
                        linestyle="--",
                        linewidth=1.5,
                        label="Average Minimum Thinking tokens",
                    )
                elif cwe == "combined":

                    total_tokens = 0
                    count = 0
                    for sub_cwe in [691, 707, 664]:
                        result = analyze_results(self.scaling_data[sub_cwe])
                        total_tokens += result["avg_tokens"]
                        count += 1
                    avg_tokens = total_tokens / count
                    ax.axvline(
                        x=avg_tokens,
                        color=LLAMA_R_COLOR,
                        linestyle="--",
                        linewidth=1.5,
                        label="Average Minimum Thinking tokens",
                    )

                all_y_values = y_values.copy()
                if "scaling" in model_metrics:
                    all_y_values.extend([m[metric] for m in model_metrics["scaling"]])
                if "par_scaling" in model_metrics:
                    all_y_values.extend(
                        [m[metric] for m in model_metrics["par_scaling"]]
                    )

                if all_y_values:
                    min_y = min(all_y_values)
                    max_y = max(all_y_values)
                    y_range = max_y - min_y

                    if y_range < 0.1:
                        y_range = 0.1

                    y_min = max(0, min_y - 0.05 * y_range)
                    y_max = min(1.0, max_y + 0.05 * y_range)
                    ax.set_ylim(y_min, y_max)

                ax.grid(True, which="major", ls="-", alpha=0.2)

                if (cwe_idx == 3) and metric_idx == 0:

                    ax.set_yticks([0.6, 0.65, 0.7])
                if (cwe_idx == 2) and metric_idx == 0:

                    ax.set_yticks([0.6, 0.65])

                if cwe_idx == 0:

                    ax.set_ylabel("")

                    if metric_idx == 0:

                        fig.text(
                            0.025,
                            0.52,
                            "Accuracy",
                            fontproperties=bold_font,
                            fontsize=9,
                            rotation=90,
                            ha="center",
                            va="center",
                        )
                    elif metric_idx == 1:

                        fig.text(
                            0.355,
                            0.52,
                            "Precision",
                            fontproperties=bold_font,
                            fontsize=9,
                            rotation=90,
                            ha="center",
                            va="center",
                        )
                    elif metric_idx == 2:

                        fig.text(
                            0.685,
                            0.52,
                            "Recall",
                            fontproperties=bold_font,
                            fontsize=9,
                            rotation=90,
                            ha="center",
                            va="center",
                        )

                subplot_index = chr(97 + (metric_idx * 4) + cwe_idx)

                ax.set_xlabel(
                    f"({subplot_index}) Thinking Tokens",
                    fontproperties=bold_font,
                    fontsize=8,
                )

                if cwe == "combined":
                    title = "Aggregate"
                else:
                    title = f"CWE-{cwe}"
                ax.set_title(title, fontproperties=bold_font, fontsize=8)

                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(regular_font)
                    label.set_fontsize(6)

                ax.set_xscale("log")
                ax.set_xlim(400, 11000)

                ax.set_xticks([500, 1000, 2000, 4000, 10000])
                ax.set_xticklabels(["500", "1k", "2k", "4k", "10k"])

                ax.margins(x=0.1, y=0.1)

        handles = [
            Line2D(
                [0],
                [0],
                color=GPT_NR_COLOR,
                marker="o",
                label="o3-mini-low",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                color=GPT_R_COLOR,
                marker="o",
                label="o3-mini-medium",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                color=HIGH_COLOR,
                marker="o",
                label="o3-mini-high",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                color=QWEN_R_COLOR,
                marker="s",
                label="r1-qn-14b(Sequential)",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                color=DEEPSEEK_R_COLOR,
                marker="^",
                label="r1-qn-14b(Parallel)",
                markersize=5,
            ),
            Line2D(
                [0],
                [0],
                color=LLAMA_R_COLOR,
                linestyle="--",
                label="r1-qn-14b(Avg init. thinking tokens)",
                markersize=5,
            ),
        ]
        legend_ax = fig.add_axes([0.1, 0.03, 0.8, 0.02])
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=6,
            prop=fm.FontProperties(
                fname="/Users/x/Library/Fonts/Lato-Regular.ttf", size=8.5
            ),
            handletextpad=0.3,
            columnspacing=1.0,
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95], h_pad=0.3, w_pad=0.5)

        plt.savefig("metrics_plots/scaling.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        self.draw_metrics_vs_tokens()
