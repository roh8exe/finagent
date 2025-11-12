"""
finagent/llm/local_llm.py
-------------------------
Lightweight local LLM implementation for offline reasoning.
Uses a small Hugging Face model like Phi-3 or TinyLlama.
"""

from finagent.llm.base import LLM
from transformers import pipeline

class LocalLLM(LLM):
    def __init__(self, 
                 name="TinyLlama", 
                 model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 temperature=0.2, 
                 max_new_tokens=512):
        """
        Initialize a small local model for reasoning.
        Works offline after initial download from Hugging Face.
        """
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        print(f"🔹 Loading local model: {model_name} ...")
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype="auto",
            device_map="auto",
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        print(f"✅ {name} model loaded successfully.")

    def chat(self, system: str, user: str, images=None) -> str:
        """
        Generate a text response using the local model.
        The interface matches the abstract LLM base class.
        """
        prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>"

        output = self.pipe(prompt, do_sample=False)[0]["generated_text"]
        response = output[len(prompt):].strip()
        return response
