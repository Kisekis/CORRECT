import json
from .base_evaluator import Evaluator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
import matplotlib.font_manager as fm
from common.globals import Args


class SingleHardData:
    def __init__(self, name, hard_data, meta_data):
        self.name = name
        self.meta_data = meta_data

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


class AbnormalEvaluator(Evaluator):
    def __init__(self):
        super().__init__()

        self.regular_font = fm.FontProperties(
            fname="/Users/x/Library/Fonts/Lato-Regular.ttf"
        )

        self.model_meta_data = {}

        self.load_data("r1-7b", 7, True, "r1-qn-7b")

    def load_single_data(self, name):

        hard_path = f"../results/hard/{name}.json"

        ret = {}

        with open(hard_path, "r") as f:
            hard_data = json.load(f)

        common_keys = set(hard_data.keys())
        for key in common_keys:
            if len(hard_data[key]) == 0:
                continue
            cwe, tokens_vuln, tokens_patched = self.dataloader.get_single_data_info_key(
                key, length_on=True
            )
            cwe = cwe.replace("'", "").replace('"', "").replace(" ", "")
            top_cwe = self.dataloader.get_top_cwe(cwe.split(","))
            d = SingleHardData(
                key,
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
        plt.figure(figsize=(4, 3))

        all_data = []

        for model_name in ["r1-7b"]:
            for key, result in self.model_meta_data[model_name]["data"].items():
                vuln_length = result.meta_data["tokens_vuln"]
                patched_length = result.meta_data["tokens_patched"]

                category = "Normal" if result.context_vuln_ret != -1 else "Abnormal"
                all_data.append({"Length": vuln_length, "Category": category})

                category = "Normal" if result.context_patched_ret != -1 else "Abnormal"
                all_data.append({"Length": patched_length, "Category": category})

        df = pd.DataFrame(all_data)

        print("\nStats:")
        for category in ["Normal", "Abnormal"]:
            category_data = df[df["Category"] == category]["Length"]
            print(f"\n{category} Category:")
            print(f"Number of samples: {len(category_data)}")
            print(f"Average: {category_data.mean():.2f}")
            print(f"Median: {category_data.median():.2f}")
            print(f"Standard deviation: {category_data.std():.2f}")

        sns.violinplot(
            data=df,
            x="Category",
            y="Length",
            palette={"Normal": Args.DEEPSEEK_NR_COLOR, "Abnormal": Args.LLAMA_NR_COLOR},
        )

        plt.ylim(0, 40000)

        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels([f"{int(x/1000)}k" for x in current_values])

        plt.title(
            "Prompt Length Distribution of r1-qn-7b", fontproperties=self.regular_font
        )
        plt.ylabel("Num of Tokens", fontproperties=self.regular_font)
        plt.tight_layout()
        plt.savefig("metrics_plots/rq1-abnormal.pdf")
        plt.close()


if __name__ == "__main__":
    evaluator = AbnormalEvaluator()
    evaluator.evaluate()
