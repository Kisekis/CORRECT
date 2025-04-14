from abc import ABC, abstractmethod
from typing import List
from common.data_classes import VulnObj, Example, VulnPairWithContext
from common.logger import logger
from common.prompts import get_zero_shot_cot_prompt_with_more_context


class CodeAnalyzer(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
    
    def zeroShotCoTAnalyze(self, pair: VulnPairWithContext, is_vuln: bool, depth: int=1, context_on: bool=True, feedback_info: str=""):
        logger.info("Starting zeroShotCoTAnalyze")
        if feedback_info:
            prompt = get_zero_shot_cot_prompt_with_more_context(pair, is_vuln, depth, context_on, feedback_info)
        else:
            prompt = get_zero_shot_cot_prompt_with_more_context(pair, is_vuln, depth, context_on)
        logger.info(f"prompt: {prompt}")
        response = self.generate(prompt)
        logger.info(f"response: {response}")
        if "HAS_VUL" in response or "has_vul" in response or "YES_VUL" in response or "HAS\_VUL" in response or "HAS\\_VUL" in response:
            return 1, response
        elif "NO_VUL" in response or "no_vul" in response or "NO\\_VUL" in response or "NO\_VUL" in response:
            return 0, response
        return -1, response

