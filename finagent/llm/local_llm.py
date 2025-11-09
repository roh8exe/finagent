"""
finagent/llm/local_llm.py
-------------------------
Local mock implementation for offline or testing use.
"""

from finagent.llm.base import LLM

class LocalLLM(LLM):
    def __init__(self, name="Local-Sim", temperature=0.0):
        self.name = name
        self.temperature = temperature

    def chat(self, system: str, user: str, images=None) -> str:
        """Simulate an LLM response (useful for testing)."""
        return f"[{self.name} simulated reply]\nSystem: {system}\nUser: {user[:120]}..."
