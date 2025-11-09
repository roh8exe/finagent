import os
import pandas as pd
import yfinance as yf

def load_prices(ticker, start, end, cache_dir="data/prices"):
    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/{ticker}.csv"

    # Download fresh data (more reliable than cache)
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True)

    # Flatten MultiIndex (sometimes ('Price', 'AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] for c in df.columns]

    # Convert all columns to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    # Handle bug: all columns named after the ticker (like amzn, tsla)
    if all(ticker.lower() in c for c in df.columns):
        df.columns = ["close", "high", "low", "open", "volume"]

    # Normalize to title case
    df.columns = [c.title() for c in df.columns]

    # Ensure "Date" as datetime index
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # Save cleaned data
    df.to_csv(path)

    # Return only valid columns
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols]
