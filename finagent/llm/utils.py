"""
finagent/llm/utils.py
---------------------
Helper functions for combining FinAgent memory and LLM queries.
"""

from typing import List, Optional
from finagent.llm.base import LLM

def finagent_query(llm: LLM, query: str, memory_context: Optional[List[str]] = None) -> str:
    """
    Combine retrieved memory context with user query and get a model response.
    """
    if memory_context:
        context_text = "\n\n".join(memory_context)
        full_prompt = f"Use the following financial context to answer:\n{context_text}\n\nUser Query: {query}"
    else:
        full_prompt = query

    system_prompt = (
        "You are FinAgent, an AI financial assistant. "
        "Answer concisely, using sound financial reasoning and data interpretation."
    )
    return llm.chat(system=system_prompt, user=full_prompt)
