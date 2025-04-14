from common.data_classes import VulnPairWithContext
from analyze.code_analyzer import CodeAnalyzer
from common.dataloader import DataLoader
import os
from datetime import datetime
import json
from common.logger import logger
from tqdm import tqdm
from mode.mode_normal import construct_eval_prompt, check_eval_result
from common.dataloader import GroundTruthInfo


def exist_in_prev_res(prev_res, key, times):
    if key in prev_res:
        if len(prev_res[key]) >= times:
            return True
    return False


def process_single_half(
    pair: VulnPairWithContext,
    evaluator: CodeAnalyzer,
    ground_truth_info: GroundTruthInfo,
    response: str,
    is_vuln: bool,
):
    prompt = construct_eval_prompt(
        pair,
        ground_truth_info.commit_msg,
        ground_truth_info.cve_desc,
        ground_truth_info.cwe_id,
        response,
        is_vuln=is_vuln,
    )
    eval_res = evaluator.generate(prompt)
    return eval_res, check_eval_result(eval_res, is_vuln)


def process_batch(
    loader: DataLoader,
    evaluator: CodeAnalyzer,
    res_folder: str = "../results/par_scaling",
    timestamp: str = None,
    res_timestamp: str = None,
):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    data = loader.get_data()

    assert timestamp is not None
    with open(f"../results/par_scaling/{timestamp}.json", "r") as f:
        prev_res = json.load(f)

    for key, pair in tqdm(data.items()):
        if key not in prev_res:
            continue
        vulns = prev_res[key][0]["vuln"]
        patcheds = prev_res[key][0]["patched"]
        for vuln in vulns:
            try:
                if vuln["need_check"] == 1:
                    ground_truth_info = loader.get_ground_truth_info(pair.name)
                    eval_res, ret = process_single_half(
                        pair,
                        evaluator,
                        ground_truth_info,
                        vuln["thinking_process"] + vuln["final_response"],
                        True,
                    )
                    vuln["eval"] = {"ret": ret, "rationale": eval_res}
            except Exception as e:
                logger.error(f"Error processing pair {key}{e}")
                print("Error processing pair {key}")
                continue
        for patched in patcheds:
            try:
                if patched["need_check"] == 1:
                    ground_truth_info = loader.get_ground_truth_info(pair.name)
                    eval_res, ret = process_single_half(
                        pair,
                        evaluator,
                        ground_truth_info,
                        patched["thinking_process"] + patched["final_response"],
                        False,
                    )
                    patched["eval"] = {"ret": ret, "rationale": eval_res}
            except Exception as e:
                logger.error(f"Error processing pair {key}: {e}")
                print("Error processing pair {key}")
                continue

    if res_timestamp == "" or res_timestamp is None:
        res_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")
    else:
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(prev_res, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {result_file}")
