from analyze.code_analyzer import CodeAnalyzer
from openai import OpenAI
from typing import List
from common.logger import logger


class OpenAIAnalyzer(CodeAnalyzer):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url="",
        max_tokens=12288,
        system_prompt="",
        temperature=0,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a vulnerability detection expert specializing in identifying specific types of vulnerabilities, particularly related to the Common Weakness Enumeration (CWE) standards.",
    ) -> str:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )
        if (
            "reasoning_content" in response.choices[0].message.model_extra
            and response.choices[0].message.model_extra["reasoning_content"] != None
            and response.choices[0].message.model_extra["reasoning_content"] != ""
        ):
            return (
                response.choices[0].message.model_extra["reasoning_content"]
                + response.choices[0].message.content
            )
        return response.choices[0].message.content
