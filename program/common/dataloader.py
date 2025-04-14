import os
from datetime import datetime
import json
from common.data_classes import VulnPairWithContext
from common.load_data import load_pair_with_more_context
from common.globals import Args
from collections import Counter
import pandas as pd
from dataclasses import dataclass
from transformers import AutoTokenizer
from common.prompts import get_zero_shot_cot_prompt_with_more_context
import random
from common.cwe_analyzer import cwe_graph


@dataclass
class GroundTruthInfo:
    commit_msg: str
    cve_desc: str
    cwe_id: str


class DataLoader:
    def __init__(
        self,
        context_folder: str = Args.CONTEXT_FOLDER,
        meta_data_path: str = Args.META_DATA_PATH,
    ):
        self.context_folder = context_folder
        self.meta_data_path = meta_data_path
        self.data = {}
        self.meta_data = pd.read_csv(meta_data_path)
        self.meta_data["short_hash"] = self.meta_data["hash"].str[:8]
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
        self.must_include = []

    def load_data(self, timestamp: str = None):
        if timestamp is not None:
            history = True
            key_list = self.load_history(timestamp)
        else:
            history = False
        for file in os.listdir(self.context_folder):
            if not file.endswith(".json"):
                continue
            project = file[:-22]
            hash = file[-21:-13]
            if history and f"{project}_{hash}" not in key_list:
                continue
            pair = load_pair_with_more_context(
                self.meta_data_path,
                f"{self.context_folder}/{project}_{hash}_context.json",
                project,
                hash,
            )
            if pair is None or pair.vuln is None or len(pair.vuln) == 0:
                continue
            self.data[f"{project}_{hash}"] = pair
        return self.data

    def dump_history(self, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        key_list = [key for key in self.data.keys()]
        with open(os.path.join(Args.HISTORY_FOLDER, f"{timestamp}.json"), "w") as f:
            json.dump(key_list, f)

    def load_history(self, timestamp: str):
        with open(os.path.join(Args.HISTORY_FOLDER, f"{timestamp}.json"), "r") as f:
            key_list = json.load(f)
        return key_list

    def print_data_info(self):
        cwe_counter = Counter()
        for key in self.data:
            for cwe in self.data[key].cwe:
                cwe_counter[cwe] += 1

        top_cwe_counter = Counter()
        for key in self.data:
            top_cwe = self.get_top_cwe(self.data[key].cwe)
            top_cwe_counter[top_cwe] += 1

    def get_data(self):
        return self.data

    def get_single_data_info_key(self, key: str, length_on: bool = False):
        repo_name = key[:-9]
        short_hash = key[-8:]
        return self.get_single_data_info(repo_name, short_hash, length_on)

    def get_single_data_info(
        self, repo_name: str, short_hash: str, length_on: bool = False
    ):
        if f"{repo_name}_{short_hash}" not in self.data:
            return None, None
        row = self.meta_data[self.meta_data["short_hash"] == short_hash].iloc[0]
        cwe_id = row["cwe_list"]
        if length_on:
            prompt1 = get_zero_shot_cot_prompt_with_more_context(
                self.data[f"{repo_name}_{short_hash}"], True, 2, True, ""
            )
            prompt2 = get_zero_shot_cot_prompt_with_more_context(
                self.data[f"{repo_name}_{short_hash}"], False, 2, True, ""
            )
            token_length1 = self.get_token_length(prompt1)
            token_length2 = self.get_token_length(prompt2)
            avg_token_length = (token_length1 + token_length2) / 2
        else:
            token_length1 = 0
            token_length2 = 0
            avg_token_length = 0
        return cwe_id, token_length1, token_length2

    def get_ground_truth_info(self, short_hash: str):
        row = self.meta_data[self.meta_data["short_hash"] == short_hash].iloc[0]
        commit_msg = row["commit_msg"]
        cve_desc = row["cve_desc"]
        cwe_id = row["cwe_list"]
        return GroundTruthInfo(commit_msg, cve_desc, cwe_id)

    def get_token_length(self, text: str):
        return len(self.tokenizer.encode(text))

    def filter_token_less_than(self, token_length: int):
        keys_to_remove = []

        for key, value in self.data.items():
            prompt = get_zero_shot_cot_prompt_with_more_context(
                value, True, 2, True, ""
            )
            token_len = self.get_token_length(prompt)
            if token_len >= token_length:
                keys_to_remove.append(key)
                print(f"{key} is removed, token length: {token_len}")

        for key in keys_to_remove:
            self.data.pop(key)

        return self.data

    def filter_history(self, timestamp: str):
        key_list = self.load_history(timestamp)
        for key in key_list:
            if key in self.data:
                self.data.pop(key)
        return self.data

    def filter_sample_cwes(self, max_sample: int = 5):
        cwe_counter = Counter()
        keys_to_remove = []
        for key in self.data:
            for cwe in self.data[key].cwe:
                if cwe_counter[cwe] < max_sample:
                    cwe_counter[cwe] += 1
                else:
                    keys_to_remove.append(key)
                    break
        for key in keys_to_remove:
            self.data.pop(key)
        return self.data

    def get_top_cwe(self, cwe_list: list[str]):
        cwes = []
        for cwe in cwe_list:
            cwe_id = int(cwe.split("-")[1])
            top_cwe = cwe_graph.top_cwe(cwe_id)
            cwes.append(top_cwe)

        if 693 in cwes:
            return 693
        if 697 in cwes:
            return 697
        if 435 in cwes:
            return 435
        return cwes[0]

    def get_single_cwe(self, key: str):
        return self.data[key].cwe[0]

    def get_repo(self, key: str):
        return key[:-9]

    def get_short_hash(self, key: str):
        return key[-8:]

    def get_function_length(self, key: str):
        vuln_functions = self.data[key].vuln
        patched_functions = self.data[key].vuln
        tokens = 0
        count = 0
        for func in vuln_functions:
            tokens += self.get_token_length(func.raw_code)
            count += 1
        for func in patched_functions:
            tokens += self.get_token_length(func.raw_code)
            count += 1
        return tokens / count

    def get_context_length(self, key: str):
        return self.get_token_length(str(self.data[key].context))

    def filter_top_sample_cwes(self, max_sample: int = 5):
        cwe_counter = Counter()

        keys_to_remove = []

        for key in self.must_include:
            top_cwe = self.get_top_cwe(self.data[key].cwe)
            if cwe_counter[top_cwe] < max_sample:
                cwe_counter[top_cwe] += 1
            else:
                print(f"{key} is removed, cwe: {top_cwe}")
                keys_to_remove.append(key)

        for key in self.data:
            if key in self.must_include:
                continue
            top_cwe = self.get_top_cwe(self.data[key].cwe)
            if cwe_counter[top_cwe] < max_sample:
                cwe_counter[top_cwe] += 1
            else:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.data.pop(key)
        return self.data

    def must_include_method(self, timestamp: str):
        key_list = self.load_history(timestamp)
        for key in key_list:
            if key in self.data:
                self.must_include.append(key)
        return self.data

    def filter_1000(self):
        keys_to_remove = []
        for key in self.data:
            is_del = True
            top_cwe = self.get_top_cwe(self.data[key].cwe)
            if top_cwe in [284, 435, 664, 682, 691, 693, 697, 703, 707, 710]:
                is_del = False
            if is_del:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.data.pop(key)
        return self.data

    def filter_one_top_cwe(self, top_cwe: int):
        keys_to_remove = []
        for key in self.data:
            top_cwe_ = self.get_top_cwe(self.data[key].cwe)
            if top_cwe_ != top_cwe:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self.data.pop(key)
        return self.data

    def sample_k_data(self, k: int):
        keys = random.sample(list(self.data.keys()), k)
        return {key: self.data[key] for key in keys}
