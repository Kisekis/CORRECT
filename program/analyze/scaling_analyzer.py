from analyze.code_analyzer import CodeAnalyzer
from common.data_classes import VulnObj, Example
from common.prompts import (
    get_zero_shot_cot_prompt_with_more_context,
)
from common.utils import format_cot_examples, format_few_shot_examples
from openai import OpenAI
from typing import List
from common.logger import logger
import requests
import json
from common.data_classes import VulnPairWithContext

import random


def check(arr, m, a1, a2, mod1, mod2):
    n = len(arr)
    aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
    h1, h2 = 0, 0
    for i in range(m):
        h1 = (h1 * a1 + arr[i]) % mod1
        h2 = (h2 * a2 + arr[i]) % mod2
    seen = {(h1, h2)}
    for start in range(1, n - m + 1):
        h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
        h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
        if (h1, h2) in seen:
            return start
        seen.add((h1, h2))
    return -1


def longestDupSubstring(s: str) -> str:
    a1, a2 = random.randint(26, 100), random.randint(26, 100)
    mod1, mod2 = random.randint(10**9 + 7, 2**31 - 1), random.randint(
        10**9 + 7, 2**31 - 1
    )
    n = len(s)
    arr = [ord(c) - ord("a") for c in s]
    l, r = 1, n - 1
    length, start = 0, -1
    while l <= r:
        m = l + (r - l + 1) // 2
        idx = check(arr, m, a1, a2, mod1, mod2)
        if idx != -1:
            l = m + 1
            length = m
            start = idx
        else:
            r = m - 1
    return s[start : start + length] if start != -1 else ""


def check_rep(s):
    return len(longestDupSubstring(s)) > 5000


class ScalingAnalyzer(CodeAnalyzer):
    def __init__(
        self,
        system_prompt="",
        min_thinking_tokens=10000,
        max_output_tokens=4096,
        temperature=0,
        max_swaps=-1,
    ):
        self.system_prompt = system_prompt
        self.min_thinking_tokens = min_thinking_tokens
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_swaps = max_swaps
        self.history = None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a vulnerability detection expert specializing in identifying specific types of vulnerabilities, particularly related to the Common Weakness Enumeration (CWE) standards.",
    ) -> str:
        if self.system_prompt != "":
            sp = self.system_prompt
        else:
            sp = system_prompt

        url = "http://localhost:8000/chat"

        if (
            self.history != None
            and self.history["thinking_tokens"] >= self.min_thinking_tokens
        ):
            return []

        if self.history != None and check_rep(self.history["thinking_process"]):
            return [
                {
                    "swap_count": 1000000,
                    "thinking_process": "",
                    "final_response": "",
                    "wait_response": "",
                    "thinking_tokens": 10000000,
                }
            ]
        if self.history != None:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "min_thinking_tokens": self.min_thinking_tokens,
                "max_output_tokens": 4096,
                "temperature": 0,
                "max_swaps": -1,
                "history": self.history,
            }
        else:
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "min_thinking_tokens": self.min_thinking_tokens,
                "max_output_tokens": 4096,
                "temperature": 0,
                "max_swaps": -1,
            }
        response = requests.post(url, json=payload)
        return response.json()

    def load_history(self, history):
        self.history = history

    def zeroShotCoTAnalyze(
        self,
        pair: VulnPairWithContext,
        is_vuln: bool,
        depth: int = 1,
        context_on: bool = True,
        feedback_info: str = "",
    ):
        logger.info("Starting zeroShotCoTAnalyze")
        if feedback_info:
            prompt = get_zero_shot_cot_prompt_with_more_context(
                pair, is_vuln, depth, context_on, feedback_info
            )
        else:
            prompt = get_zero_shot_cot_prompt_with_more_context(
                pair, is_vuln, depth, context_on
            )
        logger.info(f"prompt: {prompt}")

        response = self.generate(prompt)
        return -1, response
