"""
finagent/llm/base.py
--------------------
Abstract base interface for all LLM backends in FinAgent.
"""

from abc import ABC, abstractmethod

class LLM(ABC):
    """Abstract base class defining the FinAgent LLM interface."""

    @abstractmethod
    def chat(self, system: str, user: str, images=None) -> str:
        """
        Perform a single chat completion with the LLM.
        Args:
            system (str): System or instruction prompt.
            user (str): User input message.
            images (list, optional): Image inputs for multimodal models.
        Returns:
            str: Model response text.
        """
        pass
