from __future__ import annotations
import pandas as pd
import yfinance as yf

def fetch_adj_close(tickers: list[str], start: str = "2005-01-01") -> pd.DataFrame:
    """Fetch adjusted close via yfinance and return a DataFrame of prices."""
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    # yfinance returns multi-index columns when multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Adj Close"].copy()
    else:
        px = df.rename(columns={"Adj Close": tickers[0]})[["Adj Close"]]
    px = px.dropna(how="all")
    return px

def to_monthly_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    px_m = adj_close.resample("M").last()
    rets = px_m.pct_change().dropna(how="all")
    return rets
