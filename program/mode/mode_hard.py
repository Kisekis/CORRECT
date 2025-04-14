from common.data_classes import VulnPairWithContext
from analyze.code_analyzer import CodeAnalyzer
from common.dataloader import DataLoader
import os
from datetime import datetime
import json
from common.logger import logger
from common.dataloader import GroundTruthInfo
from mode.mode_normal import construct_eval_prompt, check_eval_result
from tqdm import tqdm
import time
from typing import Dict
from analyze.o3_analyzer import O3Analyzer

def summarize_rationale(rationale: str, evaluator: CodeAnalyzer):
    prompt = f"""
    You are a vulnerability expert. 
    You are given a rationale of a vulnerability analysis. 
    Please summarize the rationale detailly, it should include causes for vulnerabilities detected, if the locations of vulnerabilities are mentioned, please also include them.
    {rationale}
    Output format:
    <Cause> ... </Cause>
    <Cause> ... </Cause>
    ...
    """
    return evaluator.generate(prompt)


def feedback(pair: VulnPairWithContext, summary_rationale: str, ground_truth_info: GroundTruthInfo, detector: CodeAnalyzer, evaluator: CodeAnalyzer):
    """
    Implements feedback loop for false positive cases in patched code analysis
    """
    history = [{'summary_rationale': summary_rationale}]
    k = 1
    max_attempts = 5
    all_feedback = [summary_rationale]
    
    while k < max_attempts:
        # Create new prompt with previous rationale
        new_ret, new_rationale = detector.zeroShotCoTAnalyze(pair, is_vuln=False, depth=2, context_on=True, feedback_info='\n'.join(all_feedback))
        if isinstance(detector, O3Analyzer):
            reasoning_tokens = detector.get_reasoning_tokens()
        else:
            reasoning_tokens = 0
        history.append({'rationale': new_rationale, 'reasoning_tokens': reasoning_tokens})

        if new_ret == 0:
            return 0, k, history
        

            # Evaluate the new analysis
        eval_prompt = construct_eval_prompt(pair, ground_truth_info.commit_msg, 
                                    ground_truth_info.cve_desc, 
                                    ground_truth_info.cwe_id, 
                                    new_rationale, is_vuln=False)
        new_eval_res = evaluator.generate(eval_prompt)
        new_eval_ret = check_eval_result(new_eval_res, is_vuln=False)
        history.append({'eval_res': new_eval_res})

        if new_eval_ret == 1:
            return 1, k, history
        
        if new_eval_ret == 0:
            new_summary_rationale = summarize_rationale(new_rationale, evaluator)
            all_feedback.append(new_summary_rationale)
            history.append({'summary_rationale': new_summary_rationale})
            
        k += 1
    
    return 0, k, history

def add_feedback(pair: VulnPairWithContext, history: Dict[str, str], ground_truth_info: GroundTruthInfo, detector: CodeAnalyzer, evaluator: CodeAnalyzer):
    pass

def process_single(pair: VulnPairWithContext, detector: CodeAnalyzer, evaluator: CodeAnalyzer, ground_truth_info: GroundTruthInfo, ret_vuln: int=-1, response_vuln: str="", ret_patched: int=-1, response_patched: str=""):
    
    if ret_vuln == -1 and response_vuln == "":
        ret_vuln, response_vuln = detector.zeroShotCoTAnalyze(pair, is_vuln=True, depth=2, context_on=True)
    if isinstance(detector, O3Analyzer):
        reasoning_tokens_vuln = detector.get_reasoning_tokens()
    else:
        reasoning_tokens_vuln = 0
    if ret_patched == -1 and response_patched == "":
        ret_patched, response_patched = detector.zeroShotCoTAnalyze(pair, is_vuln=False, depth=2, context_on=True)
    if isinstance(detector, O3Analyzer):
        reasoning_tokens_patched = detector.get_reasoning_tokens()
    else:
        reasoning_tokens_patched = 0
    revised_ret_patched = -1
    k = 0
    history = []

    if ret_vuln == 0 or ret_vuln == -1:
        prompt_vuln = ""
        eval_res_vuln = ""
        ret_vuln_eval = -1
    elif ret_vuln == 1:
        prompt_vuln = construct_eval_prompt(pair, ground_truth_info.commit_msg, ground_truth_info.cve_desc, ground_truth_info.cwe_id, response_vuln, is_vuln=True)
        eval_res_vuln = evaluator.generate(prompt_vuln)
        ret_vuln_eval = check_eval_result(eval_res_vuln, is_vuln=True)

    if ret_patched == 0 or ret_patched == -1:
        prompt_patched = ""
        eval_res_patched = ""
        ret_patched_eval = -1
    elif ret_patched == 1:
        prompt_patched = construct_eval_prompt(pair, ground_truth_info.commit_msg, ground_truth_info.cve_desc, ground_truth_info.cwe_id, response_patched, is_vuln=False)
        eval_res_patched = evaluator.generate(prompt_patched)
        ret_patched_eval = check_eval_result(eval_res_patched, is_vuln=False)


    if ret_patched_eval == 0:
        summary_rationale = summarize_rationale(response_patched, evaluator)
        revised_ret_patched, k, history = feedback(pair, summary_rationale, ground_truth_info, detector, evaluator)
    logger.info(f"revised_ret_patched: {revised_ret_patched}, k: {k}, history: {history}")
    return ret_vuln, ret_patched, response_vuln, response_patched, eval_res_vuln, eval_res_patched, ret_vuln_eval, ret_patched_eval, revised_ret_patched, k, history, reasoning_tokens_vuln, reasoning_tokens_patched


def exist_in_prev_res(prev_res, key, times):
    if key not in prev_res or prev_res[key] is None or len(prev_res[key]) < times:
        return False
    if 'Request failed' in prev_res[key][-1]['vuln']['cot']['output']:
        return False
    if 'Request failed' in prev_res[key][-1]['patched']['cot']['output']:
        return False
    return True

def process_batch(loader: DataLoader, detector: CodeAnalyzer, evaluator: CodeAnalyzer, res_folder: str="../results/hard", times: int=1, timestamp: str=None, res_timestamp: str=None, re_evaluate: bool=False, existing_timestamp: str=None):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    data = loader.get_data()
    res_dict = {}
    
    if timestamp is not None:
        with open(f"../results/hard/{timestamp}.json", "r") as f:
            prev_res = json.load(f)
    else:
        prev_res = {}
    
    if existing_timestamp is not None:
        with open(f"../results/hard/{existing_timestamp}.json", "r") as f:
            existing_res = json.load(f)
    else:
        existing_res = {}
    
    for key, pair in tqdm(data.items()):
        if exist_in_prev_res(prev_res, key, times) and not re_evaluate:
            res_dict[key] = prev_res[key]
            continue
        if exist_in_prev_res(prev_res, key, times) and re_evaluate:
            if prev_res[key][-1]['vuln']['eval']['ret'] == -1 and prev_res[key][-1]['patched']['eval']['ret'] == -1:
                res_dict[key] = prev_res[key]
                continue
            if key in existing_res and existing_res[key] is not None and len(existing_res[key]) > 0:
                res_dict[key] = existing_res[key]
                continue
        res_dict[key] = []
        for i in range(times):
            try:
                ground_truth_info = loader.get_ground_truth_info(pair.name)
                if exist_in_prev_res(prev_res, key, times) and re_evaluate:
                    ret_vuln, ret_patched, response_vuln, response_patched, eval_res_vuln, eval_res_patched, ret_vuln_eval, ret_patched_eval, revised_ret_patched, k, history, reasoning_tokens_vuln, reasoning_tokens_patched = process_single(pair, detector, evaluator, ground_truth_info, prev_res[key][-1]['vuln']['cot']['ret'], prev_res[key][-1]['vuln']['cot']['output'], prev_res[key][-1]['patched']['cot']['ret'], prev_res[key][-1]['patched']['cot']['output'])
                else:
                    ret_vuln, ret_patched, response_vuln, response_patched, eval_res_vuln, eval_res_patched, ret_vuln_eval, ret_patched_eval, revised_ret_patched, k, history, reasoning_tokens_vuln, reasoning_tokens_patched = process_single(pair, detector, evaluator, ground_truth_info)
                res_dict[key].append({
                    'vuln': {
                        'cot': {'ret': ret_vuln, 'output': response_vuln, 'reasoning_tokens': reasoning_tokens_vuln},
                        'eval': {'ret': ret_vuln_eval, 'rationale': eval_res_vuln},
                    },
                    'patched': {
                        'cot': {'ret': ret_patched, 'output': response_patched, 'reasoning_tokens': reasoning_tokens_patched},
                        'eval': {'ret': ret_patched_eval, 'rationale': eval_res_patched},
                        'feedback': {'ret': revised_ret_patched, 'k': k, 'history': history}
                    }
                })
            except Exception as e:
                print(e)
                logger.error(f"Error processing pair {key} (attempt {i+1}/{times}): {e}")
                continue
    
    if res_timestamp == '' or res_timestamp is None:
        res_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")
    else:
        result_file = os.path.join(res_folder, f"result_{res_timestamp}.json")
    
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(res_dict, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {result_file}")
