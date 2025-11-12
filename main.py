# main.py
import os, json
import pandas as pd
from datetime import timedelta
from config import CFG
from finagent.llm.local_llm import LocalLLM
from finagent.llm.utils import finagent_query
from finagent.embed.encoder import TextEncoder
from finagent.embed.store import VectorStore
from finagent.dataio.prices import load_prices
from finagent.dataio.charts import render_kline
from finagent.modules.market_intel import MarketIntelligence
from finagent.modules.low_reflection import LowReflection
from finagent.modules.high_reflection import HighReflection
from finagent.modules.decision import Decision
from finagent.eval.backtest import run_backtest
from finagent.eval.metrics import arr, sharpe, volatility, max_drawdown, calmar, sortino

def run_for_ticker(ticker):
    df = load_prices(ticker, CFG.train_start, CFG.test_end)
    df = df[(df.index >= CFG.train_start) & (df.index < CFG.test_end)]
    encoder = TextEncoder(CFG.embed_model)
    store = VectorStore(dim=encoder.embed(["x"]).shape[1])
    llm = LocalLLM(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = {
        "mi": "finagent/prompts/market_intel.txt",
        "llr": "finagent/prompts/low_reflection.txt",
        "hlr": "finagent/prompts/high_reflection.txt",
        "decision": "finagent/prompts/decision.txt",
    }
    MI  = MarketIntelligence(llm, encoder, store, prompts)
    LLR = LowReflection(llm, encoder, store, prompts)
    HLR = HighReflection(llm, encoder, store, prompts)
    DEC = Decision(llm, prompts, trader_pref="balanced")

    # TRAIN PHASE: build memory on [train_start, train_end)
    train_df = df[(df.index >= CFG.train_start) & (df.index < CFG.train_end)]
    for dt, row in train_df.iterrows():
        # MI
        if CFG.ablations["use_MI"]:
            mi_out = MI.latest(ticker, dt, train_df.loc[:dt].tail(60))
            MI.write_to_memory(ticker, dt, mi_out)


    # TEST PHASE: daily decisions
    test_df = df[(df.index >= CFG.test_start) & (df.index < CFG.test_end)]
    decisions = {}
    action_log = []
    equity_curve_stats = {}

    for dt, row in test_df.iterrows():
        window_df = df.loc[:dt].tail(120)              # recent window
        chart_path = f"data/charts/{ticker}_{str(dt.date())}.png"
        render_kline(window_df, chart_path, title=f"{ticker} {str(dt.date())}")

        latest_mi_summary = ""
        llr_reasoning = ""
        hlr_summary = ""

        if CFG.ablations["use_MI"]:
            mi_out = MI.latest(ticker, dt, window_df)
            MI_item = MI.write_to_memory(ticker, dt, mi_out)
            latest_mi_summary = MI_item.summary
            retrieved = MI.diversified_retrieve(ticker, mi_out)
        else:
            retrieved = []

        if CFG.ablations["use_LLR"]:
            llr_js, llr_item = LLR.run(
                ticker, dt, latest_mi_summary, retrieved, chart_path
            )
            llr_reasoning = llr_js.get("reasoning","")

        if CFG.ablations["use_HLR"]:
            recent_decisions = action_log[-14:]
            hlr_js, hlr_item = HLR.run(
                ticker, dt, recent_decisions, chart_path, equity_curve_stats
            )
            hlr_summary = hlr_js.get("summary","")

        use_tools = CFG.ablations["use_Tools"]
        dec_js, tools = DEC.run(
            ticker, dt,
            latest_mi_summary=latest_mi_summary,
            llr_reasoning=llr_reasoning,
            hlr_summary=hlr_summary,
            df=window_df,
            use_tools=use_tools
        )
        act = dec_js.get("action","HOLD").upper()
        if act not in ("BUY","SELL","HOLD"): act = "HOLD"
        decisions[dt] = act
        action_log.append({"date": str(dt.date()), "action": act, "reason": dec_js.get("reasoning","")})

    equity_np, equity_series = run_backtest(test_df, decisions, initial_cash=CFG.initial_cash,
                                            commission_bps=CFG.commission_bps)
    metrics = {
        "ARR": arr(equity_np), "Sharpe": sharpe(equity_np),
        "Sortino": sortino(equity_np), "Calmar": calmar(equity_np),
        "Vol": volatility(equity_np), "MDD": max_drawdown(equity_np)
    }
    return decisions, equity_series, metrics

if __name__ == "__main__":
    results = {}
    for t in CFG.tickers:
        print(f"=== {t} ===")
        dec, eq, m = run_for_ticker(t)
        print(m)

        #Save predictions for each ticker
        import pandas as pd
        pd.DataFrame([
            {"date": d.date(), "action": a}
            for d, a in dec.items()
        ]).to_csv(f"data/decisions_{t}.csv", index=False)
        print(f" Saved predictions for {t} to data/decisions_{t}.csv")

        results[t] = {"metrics": m}
