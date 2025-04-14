from common.data_classes import VulnPairWithContext
from analyze.code_analyzer import CodeAnalyzer
from common.dataloader import DataLoader
import os
from datetime import datetime
import json
from common.logger import logger
from tqdm import tqdm


def exist_in_prev_res(prev_res, key, times):
    if key in prev_res:
        if len(prev_res[key]) >= times:
            return True
    return False


def process_single(
    pair: VulnPairWithContext, analyzer: CodeAnalyzer, history_vuln, history_patched
):
    analyzer.load_history(history_vuln)
    ret_vuln, response_vuln = analyzer.zeroShotCoTAnalyze(
        pair, is_vuln=True, depth=2, context_on=True
    )
    analyzer.load_history(history_patched)
    ret_patched, response_patched = analyzer.zeroShotCoTAnalyze(
        pair, is_vuln=False, depth=2, context_on=True
    )
    return ret_vuln, ret_patched, response_vuln, response_patched


def process_batch(
    loader: DataLoader,
    analyzer: CodeAnalyzer,
    res_folder: str = "../results/seq_scaling",
    times: int = 1,
    timestamp: str = None,
    res_timestamp: str = None,
):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    data = loader.get_data()
    res_dict = {}

    if timestamp is not None:
        with open(f"../results/seq_scaling/{timestamp}.json", "r") as f:
            prev_res = json.load(f)
    else:
        prev_res = {}

    for key, pair in tqdm(data.items()):
        res_dict[key] = []
        if exist_in_prev_res(prev_res, key, times):
            res_dict[key] = prev_res[key]
        for i in range(times):
            try:
                history_vuln = history_patched = None
                if key in res_dict and len(res_dict[key]) > 0:
                    history_vuln = res_dict[key][-1]["vuln"][-1]
                    history_patched = res_dict[key][-1]["patched"][-1]
                ret_vuln, ret_patched, response_vuln, response_patched = process_single(
                    pair, analyzer, history_vuln, history_patched
                )

                if key in res_dict and len(res_dict[key]) > 0:
                    res_dict[key][-1]["vuln"] += response_vuln
                    res_dict[key][-1]["patched"] += response_patched
                else:
                    res_dict[key].append(
                        {"vuln": response_vuln, "patched": response_patched}
                    )
            except Exception as e:
                logger.error(
                    f"Error processing pair {key} (attempt {i+1}/{times}): {e}"
                )
                continue

    if res_timestamp == "" or res_timestamp is None:
        res_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")
    else:
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {result_file}")
