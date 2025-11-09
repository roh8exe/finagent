"""
finagent/llm/openai_llm.py
--------------------------
Concrete implementation of LLM using OpenAI ChatCompletion API.
"""

import os
from openai import OpenAI
from finagent.llm.base import LLM

class OpenAILLM(LLM):
    def __init__(self, model="gpt-4o-mini", temperature=0.3, max_tokens=800):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, system: str, user: str, images=None) -> str:
        """Send a chat completion request to OpenAI."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        # Optional multimodal support
        if images:
            messages[-1]["content"] = [
                {"type": "text", "text": user},
                *[{"type": "image_url", "image_url": img} for img in images]
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
