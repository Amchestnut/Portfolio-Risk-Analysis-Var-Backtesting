from __future__ import annotations
import numpy as np
import pandas as pd

def normalize_weights(weights: dict[str, float]) -> pd.Series:
    """
    We just want to normalize the weights so that they sum up to 1.
    If we have LEVERAGE, we want to "neutralize" it, so we dont take it into account.
    This is done by normalizing the weights so that they sum up to 1.
    """
    w = pd.Series(weights, dtype=float)
    s = w.sum()
    if not np.isclose(s, 1.0):
        w = w / s
    return w

def portfolio_returns(ret_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    //// r_p,t = SUM_i (w_i * r_i,t) ////
    R_pt = portfolio returns in moment "t", more precise "at day t", summed up
    w = instrument weight
    r_it = share of the instrument "i" in moment "t".

    .sum(axis=1) adds up the information per ROW, and ignores NaN values. Example: 2024-01-02: added up to 1.4%
    Output: pd.Series, index=dates(t),  values=(porfolio return R_pt)
    """
    w = normalize_weights(weights)
    # reindex to existing columns; fill missing weights (NaN) as 0.0
    w = w.reindex(ret_df.columns).fillna(0.0)
    return (ret_df * w).sum(axis=1)