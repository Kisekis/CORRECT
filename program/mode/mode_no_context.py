from common.data_classes import VulnPairWithContext
from analyze.code_analyzer import CodeAnalyzer
from common.dataloader import DataLoader
import os
from datetime import datetime
import json
from common.logger import logger
from tqdm import tqdm
from analyze.o3_analyzer import O3Analyzer
def process_single(pair: VulnPairWithContext, analyzer: CodeAnalyzer):
    ret_vuln, response_vuln = analyzer.zeroShotCoTAnalyze(pair, is_vuln=True, depth=2, context_on=False)
    if isinstance(analyzer, O3Analyzer):
        reasoning_tokens_vuln = analyzer.get_reasoning_tokens()
    else:
        reasoning_tokens_vuln = 0
    ret_patched, response_patched = analyzer.zeroShotCoTAnalyze(pair, is_vuln=False, depth=2, context_on=False)
    if isinstance(analyzer, O3Analyzer):
        reasoning_tokens_patched = analyzer.get_reasoning_tokens()
    else:
        reasoning_tokens_patched = 0
    return ret_vuln, ret_patched, response_vuln, response_patched, reasoning_tokens_vuln, reasoning_tokens_patched

def exist_in_prev_res(prev_res, key, times):
    if key not in prev_res or prev_res[key] is None or len(prev_res[key]) < times:
        return False
    if 'Request failed' in prev_res[key][-1]['vuln']['cot']['output']:
        return False
    if 'Request failed' in prev_res[key][-1]['patched']['cot']['output']:
        return False
    return True
def process_batch(loader: DataLoader, analyzer: CodeAnalyzer, res_folder: str="../results/no_context", times: int=1, timestamp: str=None, res_timestamp: str=None):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    data = loader.get_data()
    res_dict = {}
    
    if timestamp is not None:
        with open(f"../results/no_context/{timestamp}.json", "r") as f:
            prev_res = json.load(f)
    else:
        prev_res = {}
    for key, pair in tqdm(data.items()):
        if exist_in_prev_res(prev_res, key, times):
            res_dict[key] = prev_res[key]
            continue
        res_dict[key] = []
        for i in range(times):
            try:
                ret_vuln, ret_patched, response_vuln, response_patched, reasoning_tokens_vuln, reasoning_tokens_patched = process_single(pair, analyzer)
                res_dict[key].append({
                    'vuln': {
                        'cot': {'ret': ret_vuln, 'output': response_vuln, 'reasoning_tokens': reasoning_tokens_vuln},
                    },
                    'patched': {
                        'cot': {'ret': ret_patched, 'output': response_patched, 'reasoning_tokens': reasoning_tokens_patched},
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
