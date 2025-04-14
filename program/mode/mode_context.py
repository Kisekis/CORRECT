from common.data_classes import VulnPairWithContext
from analyze.code_analyzer import CodeAnalyzer
from common.dataloader import DataLoader
import os
from datetime import datetime
import json
from common.logger import logger
from tqdm import tqdm


def process_single(pair: VulnPairWithContext, analyzer: CodeAnalyzer):
    ret_vuln, response_vuln = analyzer.zeroShotCoTAnalyze(pair, is_vuln=True, depth=2, context_on=True)
    ret_patched, response_patched = analyzer.zeroShotCoTAnalyze(pair, is_vuln=False, depth=2, context_on=True)
    return ret_vuln, ret_patched, response_vuln, response_patched

def process_batch(loader: DataLoader, analyzer: CodeAnalyzer, res_folder: str="../results/context", timestamp: str=None, res_timestamp: str=None, times: int=1):
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    
    data = loader.get_data()
    res_dict = {}
    
    if timestamp is not None:
        with open(f"../results/context/{timestamp}.json", "r") as f:
            res_dict = json.load(f)
    
    for key, pair in tqdm(data.items()):
        if key not in res_dict:
            res_dict[key] = []
        for i in range(times):
            try:
                ret_vuln, ret_patched, response_vuln, response_patched = process_single(pair, analyzer)
                res_dict[key].append({
                    'vuln': {
                        'cot': {'ret': ret_vuln, 'output': response_vuln}
                    },
                    'patched': {
                        'cot': {'ret': ret_patched, 'output': response_patched}
                    }
                })
            except Exception as e:
                logger.error(f"Error processing pair {key} (attempt {i+1}/{times}): {e}")
                print(e)
                continue
    
    if res_timestamp is not None:
        with open(f"../results/context/{res_timestamp}.json", "w") as f:
            json.dump(res_dict, f, ensure_ascii=False, indent=2)
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_file = os.path.join(res_folder, f"result_{timestamp}.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w", encoding='utf-8') as f:
            json.dump(res_dict, f, ensure_ascii=False, indent=2)
    
