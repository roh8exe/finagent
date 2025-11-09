import os
import mplfinance as mpf
import pandas as pd

def render_kline(df: pd.DataFrame, out_path: str, title: str = ""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mpf.plot(df, type="candle", volume=True, style="yahoo",
             title=title,
             savefig=dict(fname=out_path, dpi=120, bbox_inches="tight"))
    return out_path
