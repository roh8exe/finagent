# test_llm.py

from finagent.llm.local_llm import LocalLLM
from finagent.llm.utils import finagent_query

# Initialize Local mock model (offline)
llm = LocalLLM(name="Offline-FinAgent")

# Add some dummy memory context (optional)
context = [
    "AAPL showed strong bullish momentum in January 2023 with RSI > 70.",
    "MACD crossed above signal line indicating upward trend."
]

# Ask a test query
query = "Based on this context, should I consider AAPL overbought?"

response = finagent_query(llm, query, context)
print("\n--- Local LLM Output ---")
print(response)
