from dataclasses import dataclass

@dataclass
class Settings:
    tickers = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "ETH-USD"]
    train_start = "2022-06-01"
    train_end   = "2023-06-01"
    test_start  = "2023-06-02"
    test_end    = "2024-01-01"
    initial_cash = 100_000
    commission_bps = 1   # 1 bp per trade
    chart_img_size = (900, 500)
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "gpt-4o-mini"   # or your local model
    ablations = dict(
        use_MI=True, use_LLR=True, use_HLR=True, use_Tools=True
    )

CFG = Settings()
