from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Sequence, Optional
import yfinance as yf


def load_prices_yf(tickers: Sequence[str], start: str = "2005-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """
    Tickers are liquid instruments
    Preuzima Adjusted Close (to je cena korigovana za dividente i splitove) za date instrumente (AAPL, MSFT, EURUSD=X, ...).
    Need "conda install -n py3.12 sqlite" to download
    If is series (one column of data, 1D), convert into DataFrame 2D type.

    .sort_index() is used so we can sort the dates increasingly

    returns DataFrame (Date index, columns=instruments).
    """
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all").sort_index()

def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    prices.shift(1) moves all prices 1 row down (compares Pt with Pt-1)
    rt =  0.006543 ->  equals +0.6543% that day
    rt = -0.013977 ->  equals −1.3977% that day
    Daily log returns.
    If everything in a row is NaN, clean it.
    """
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")




if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "AMZN", "EURUSD=X", "USDJPY=X"]   # 5-10 likvidnih instrumenata
    weights = {"AAPL": 0.25, "MSFT": 0.25, "AMZN": 0.25, "EURUSD=X": 0.15, "USDJPY=X": 0.10}
    alpha_levels = [0.95, 0.99]
    window = 250
    prices = load_prices_yf(tickers, start="2006-01-01")
    print(prices.head())
    rets = to_log_returns(prices)
    print(rets.head())