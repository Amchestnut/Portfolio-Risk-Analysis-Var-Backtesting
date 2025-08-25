import numpy as np
import pandas as pd
from scipy.stats import chi2

def christoffersen_independence(hits: pd.Series) -> tuple[float, float]:
    """
    Christoffersen (IND) calculates if independence are independent of each other (with NO CLUSTERING)
    YES means that the model catches the volatility dynamics
    No means that the exceptions are grouping, and the model is slow to react
    "Test of independence with 2x2 transitional matrix"

    LR_ind ~ Chi2(1)
    """
    h = hits.values.astype(int)
    if h.size < 2:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(h)):
        prev, curr = h[i - 1], h[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1

    eps = 1e-12
    n0x = n00 + n01
    n1x = n10 + n11
    pi01 = n01 / n0x if n0x > 0 else 0.0
    pi11 = n11 / n1x if n1x > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)

    # log likelihood (probabilities)
    def ln(p):
        return np.log(min(max(p, eps), 1 - eps))

    lnL1 = n00 * ln(1 - pi01) + n01 * ln(pi01) + n10 * ln(1 - pi11) + n11 * ln(pi11)
    lnL0 = (n00 + n10) * ln(1 - pi) + (n01 + n11) * ln(pi)
    LR_ind = -2 * (lnL0 - lnL1)
    pval = 1 - chi2.cdf(LR_ind, df=1)

    return float(LR_ind), float(pval)