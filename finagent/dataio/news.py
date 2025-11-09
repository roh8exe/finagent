from datetime import datetime

def fetch_news(ticker: str, dt: datetime):
    # TODO: plug real news API later
    return [f"{ticker} placeholder headline on {dt.date()}"]
