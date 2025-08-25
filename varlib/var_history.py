import numpy as np
import pandas as pd

def history_var_expected_loss(
    portfolio_returns: pd.Series,
    alpha: float = 0.95,
    window: int = 250,
    lagged: bool = True,
) -> pd.DataFrame:
    """
    This method uses historical returns to estimate the potential loss. It involves sorting the historical returns and finding the percentile that corresponds to the desired confidence level.
    We want to calculate the VaR and ES.
    portfolio_returns = daily portfolio returns
    alpha = quantile of the tail that i am calculating (0.95 or 0.99)
    window = number of days to look back (length of the sliding window)
    lagged = if True, result at the day t uses only data until t-1 (no looking ahead), meaning: numbers are predictions for the next day, not "explaining" the current day. This is the right thing to do for back-testing

    var_series = take window of x days, and calculate alpha quantile LOSS in that window.
    That's the VaR that day (before moving).
    It is just a pd.Series with VaR values, for example 0.02 = 2% daily VaR

    HS: VaR = α-quantile losses = quantile(-r).
    ES = mean(losses >= VaR) per window
    ES (expected Shortfall) is the average loss in the tail left of VaR vertical line
    """

    # losses will be represented as POSITIVE numbers Lt = -Rpt
    # (if the return is -2%, the loss is +2%)
    losses = -portfolio_returns
    var_series = losses.rolling(window).quantile(alpha, interpolation="linear")

    def _expected_shortfall_tail(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        v = np.quantile(x, alpha)
        # Take all LOSSES greater or equal than VAR, we want the average loss that was greater than VaR
        tail = x[x >= v]
        # return the mean value of this part of the tail, but if not a lot of data, return nan.
        return float(np.mean(tail)) if tail.size > 0 else np.nan

    es_series = losses.rolling(window).apply(_expected_shortfall_tail, raw=True)
    out = pd.DataFrame({"VaR": var_series, "ES": es_series})
    if lagged:
        out = out.shift(1)
    return out
