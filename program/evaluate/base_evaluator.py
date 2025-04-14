import json
from common.dataloader import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import matplotlib.font_manager as fm
from common.globals import Args


class SingleResult:
    def __init__(self, name, no_context_data, hard_data, meta_data):
        self.name = name
        self.meta_data = meta_data
        self.no_context_vuln_ret = no_context_data["vuln"]["cot"]["ret"]
        self.no_context_patched_ret = no_context_data["patched"]["cot"]["ret"]
        self.no_context_vuln_output = no_context_data["vuln"]["cot"]["output"]
        self.no_context_patched_output = no_context_data["patched"]["cot"]["output"]

        self.context_vuln_ret = hard_data["vuln"]["cot"]["ret"]
        self.context_patched_ret = hard_data["patched"]["cot"]["ret"]
        self.context_vuln_output = hard_data["vuln"]["cot"]["output"]
        self.context_patched_output = hard_data["patched"]["cot"]["output"]

        if hard_data["vuln"]["cot"]["ret"] == 1:
            self.normal_vuln_ret = int(hard_data["vuln"]["eval"]["ret"])
            self.hard_vuln_ret = int(hard_data["vuln"]["eval"]["ret"])
        elif hard_data["vuln"]["cot"]["ret"] == 0:
            self.normal_vuln_ret = 0
            self.hard_vuln_ret = 0
        else:
            self.normal_vuln_ret = -1
            self.hard_vuln_ret = -1

        if hard_data["patched"]["cot"]["ret"] == 1:
            if hard_data["patched"]["eval"]["ret"]:
                self.normal_patched_ret = 1
                self.hard_patched_ret = 1
            else:
                self.normal_patched_ret = 0
                self.hard_patched_ret = hard_data["patched"]["feedback"]["ret"]
        elif hard_data["patched"]["cot"]["ret"] == 0:
            self.normal_patched_ret = 0
            self.hard_patched_ret = 0
        else:
            self.normal_patched_ret = -1
            self.hard_patched_ret = -1

        self.normal_vuln_output = hard_data["vuln"]["cot"]["output"]
        self.normal_patched_output = hard_data["patched"]["cot"]["output"]
        self.eval_vuln_ret = hard_data["vuln"]["eval"]["ret"]
        self.eval_vuln_output = hard_data["vuln"]["eval"]["rationale"]
        self.eval_patched_ret = hard_data["patched"]["eval"]["ret"]
        self.eval_patched_output = hard_data["patched"]["eval"]["rationale"]

        self.hard_vuln_output = hard_data["vuln"]["cot"]["output"]
        self.hard_patched_output = hard_data["patched"]["cot"]["output"]
        self.hard_patched_feedback = hard_data["patched"]["feedback"]

        if (
            "reasoning_tokens" in hard_data["vuln"]["cot"]
            and hard_data["vuln"]["cot"]["reasoning_tokens"] > 0
        ):
            self.hard_vuln_reasoning_tokens = hard_data["vuln"]["cot"][
                "reasoning_tokens"
            ]
        else:
            self.hard_vuln_reasoning_tokens = 0

        if (
            "reasoning_tokens" in hard_data["patched"]["cot"]
            and hard_data["patched"]["cot"]["reasoning_tokens"] > 0
        ):
            self.hard_patched_reasoning_tokens = hard_data["patched"]["cot"][
                "reasoning_tokens"
            ]
        else:
            self.hard_patched_reasoning_tokens = 0


class Evaluator:
    def __init__(self):
        self.model_meta_data = {}
        self.dataloader = DataLoader()
        self.dataloader.load_data("CIVDataset-Small")
        self.load_data("7b", 7, False, "qn-7b")
        self.load_data("8b", 8, False, "lm-8b")
        self.load_data("14b", 14, False, "qn-14b")
        self.load_data("32b", 32, False, "qn-32b")
        self.load_data("r1-7b", 7, True, "r1-qn-7b")
        self.load_data("r1-8b", 8, True, "r1-lm-8b")
        self.load_data("r1-14b", 14, True, "r1-qn-14b")
        self.load_data("r1-32b", 32, True, "r1-qn-32b")

        self.load_data("v3", 671, False, "ds-v3")
        self.load_data("70b", 70, False, "lm-70b")
        self.load_data("r1-70b", 70, True, "r1-lm-70b")
        self.load_data("r1", 671, True, "ds-r1")
        self.load_data("o3-mini-medium", 671, True, "o3-mini")

    def drop_data(self, model_name):
        del self.model_meta_data[model_name]

    def load_data(self, model_name, k_param, is_reasoning, full_name=""):

        self.model_meta_data[model_name] = {
            "param": k_param,
            "is_reasoning": is_reasoning,
            "data": self.load_single_data(model_name),
            "full_name": full_name,
        }

    def load_single_data(self, name):
        no_context_path = f"../results/no_context/{name}.json"
        hard_path = f"../results/hard/{name}.json"

        ret = {}

        with open(hard_path, "r") as f:
            hard_data = json.load(f)

        if os.path.exists(no_context_path):
            with open(no_context_path, "r") as f:
                no_context_data = json.load(f)
        else:

            no_context_data = {}
            for key in hard_data.keys():
                if len(hard_data[key]) > 0:

                    no_context_data[key] = [
                        {
                            "vuln": {
                                "cot": {"ret": 0, "output": "Mocked no context output"}
                            },
                            "patched": {
                                "cot": {"ret": 0, "output": "Mocked no context output"}
                            },
                        }
                    ]

        common_keys = set(no_context_data.keys()) & set(hard_data.keys())
        for key in common_keys:
            if len(no_context_data[key]) * len(hard_data[key]) == 0:
                continue
            cwe, tokens_vuln, tokens_patched = self.dataloader.get_single_data_info_key(
                key
            )
            cwe = cwe.replace("'", "").replace('"', "").replace(" ", "")
            top_cwe = self.dataloader.get_top_cwe(cwe.split(","))
            d = SingleResult(
                key,
                no_context_data=no_context_data[key][0],
                hard_data=hard_data[key][0],
                meta_data={
                    "cwe": cwe,
                    "tokens_vuln": tokens_vuln,
                    "tokens_patched": tokens_patched,
                    "top-cwe": top_cwe,
                },
            )
            ret[key] = d

        return ret

    def evaluate(self):
        plt.figure(figsize=(5.5, 5))

        regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )
        bold_font = fm.FontProperties(fname="/Users/x/Library/Fonts/Lato-Bold.ttf")

        model_order = [
            "7b",
            "14b",
            "32b",
            "8b",
            "70b",
            "r1-7b",
            "r1-14b",
            "r1-32b",
            "r1-8b",
            "r1-70b",
            "o3-mini-medium",
            "v3",
            "r1",
        ]
        model_order.reverse()
        model_labels = [
            self.model_meta_data[m]["full_name"]
            for m in model_order
            if m in self.model_meta_data
        ]
        valid_models = [m for m in model_order if m in self.model_meta_data]

        result_types = ["(1,0)", "(0,0)", "(1,1)", "(0,1)", "other"]
        colors = ["#8FC0DB", "#CDE1EB", "#f4e1b4", "#ddc58f", "lightgray"]

        bar_height = 0.8

        y_positions = []
        current_pos = 0
        valid_models.reverse()
        model_labels.reverse()

        for i, model in enumerate(valid_models):
            if model == "8b":
                current_pos += 2
            elif model == "r1-7b":
                current_pos += 2
            elif model == "r1-8b":
                current_pos += 2
            elif model == "o3-mini-medium" or model == "v3":
                current_pos += 2
            elif model == "7b":
                current_pos += 0
            else:
                current_pos += 0.9
            y_positions.append(current_pos)

        y_positions.reverse()
        valid_models.reverse()
        model_labels.reverse()

        dot_positions = {}
        for model_idx, model_name in enumerate(valid_models):
            results = defaultdict(int)
            total = len(self.model_meta_data[model_name]["data"])

            for result in self.model_meta_data[model_name]["data"].values():
                pair = (result.hard_vuln_ret, result.hard_patched_ret)
                if pair == (0, 1):
                    results["(0,1)"] += 1
                elif pair == (1, 1):
                    results["(1,1)"] += 1
                elif pair == (0, 0):
                    results["(0,0)"] += 1
                elif pair == (1, 0):
                    results["(1,0)"] += 1
                else:
                    results["other"] += 1

            left = 0
            for result_type, color in zip(result_types, colors):
                width = results[result_type] / total if total > 0 else 0
                if width > 0:
                    plt.barh(
                        y_positions[model_idx],
                        width,
                        bar_height,
                        left=left,
                        color=color,
                        label=result_type if model_idx == len(valid_models) - 1 else "",
                    )
                    percentage = width * 100
                    if percentage >= 5:
                        plt.text(
                            left + width / 2,
                            y_positions[model_idx],
                            f"{percentage:.0f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            fontproperties=bold_font,
                        )
                if result_type == "(1,0)":
                    dot_positions[model_name] = (left + width, y_positions[model_idx])
                left += width

        for model_name in dot_positions:
            if model_name in ["o3-mini-medium", "v3", "r1"]:
                continue
            x, y = dot_positions[model_name]
            plt.plot(x, y, "ko", markersize=4, color="#8FC0DB")

        groups = [
            ["r1-8b", "r1-70b"],
            ["r1-32b", "r1-14b", "r1-7b"],
            ["70b", "8b"],
            ["32b", "14b", "7b"],
        ]

        for group in groups:
            x_coords = [
                dot_positions[model][0] for model in group if model in dot_positions
            ]
            y_coords = [
                dot_positions[model][1] for model in group if model in dot_positions
            ]
            plt.plot(x_coords, y_coords, "-", color="#68a5c6", linewidth=1.5)

        plt.yticks(y_positions, model_labels)
        ax = plt.gca()
        ax.tick_params(axis="y", pad=0)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(bold_font)
            if label in ax.get_yticklabels():
                label.set_fontsize(7)

        plt.xlim(0, 1)

        legend = plt.legend(
            bbox_to_anchor=(0.5, 1.08),
            loc="upper center",
            ncol=5,
            borderaxespad=0.0,
            columnspacing=1.4,
        )
        for text in legend.get_texts():
            text.set_fontproperties(regular_font)

        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.tight_layout()
        plt.savefig("metrics_plots/pair-wise.svg", bbox_inches="tight", dpi=300)
        plt.close()
