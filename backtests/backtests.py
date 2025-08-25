import numpy as np
import pandas as pd
from scipy.stats import chi2

from backtests.christoffersen_method import christoffersen_independence
from backtests.kupiec_pof import kupiec_pof


def exceedances(port_ret: pd.Series, var_series: pd.Series) -> pd.Series:
    """
    How many times we exceeded VaR?
    Hit = 1 if L_t > VaR_t (tj. -r_t > VaR_t).
    """
    aligned = pd.concat([port_ret, var_series], axis=1, keys=["r", "VaR"]).dropna()
    hits = ((-aligned["r"]) > aligned["VaR"]).astype(int)
    return hits

def lr_cc(hits: pd.Series, alpha: float) -> tuple[float, float]:
    """
    LRcc (conditional coverage) both Kupiec and Christoffersen conditions together.
    This is the main "checker", if p>0.05, good.
    """
    LR_uc, p_uc, _ = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LR_cc = LR_uc + LR_ind
    p_cc = 1 - chi2.cdf(LR_cc, df=2)
    return float(LR_cc), float(p_cc)

def basel_traffic_light(x: int, T: int, alpha: float) -> str:
    """
    Basel reference (for 99% and T≈250):
      0–4 green, 5–9 yellow, ≥10 red.
    For other T/apha no selected colors, only n/a.
    """
    if abs(alpha - 0.99) < 1e-6 and 200 <= T <= 300:
        if x <= 4: return "green"
        if x <= 9: return "yellow"
        return "red"
    return "n/a"

def summarize_backtests(port_ret: pd.Series, var_series: pd.Series, alpha: float, label: str) -> pd.DataFrame:
    hits = exceedances(port_ret, var_series)
    T = len(hits)

    LR_uc, p_uc, X = kupiec_pof(hits, alpha)
    LR_ind, p_ind = christoffersen_independence(hits)
    LRcc, p_cc = lr_cc(hits, alpha)

    row = {
        "method": label,
        "alpha": alpha,
        "T": T,
        "exceedances": X,
        "Kupiec_LR": LR_uc, "Kupiec_p": p_uc,
        "Christ_LR": LR_ind, "Christ_p": p_ind,
        "LRcc": LRcc, "LRcc_p": p_cc,
        "traffic_light": basel_traffic_light(X, T, alpha),
    }

    return pd.DataFrame([row]).set_index("method")
