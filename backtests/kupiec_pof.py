import numpy as np
import pandas as pd
from scipy.stats import chi2

def kupiec_pof(hits: pd.Series, alpha: float) -> tuple[float, float, int]:
    """
    Kupiec (POF) calculates if the frequency of exceptions is exactly p = 1 - α
    If YES, the quantile is (in average) calibrated
    if NO, it means that the VaR is too high (too little exceptions) or too low (too many exceptions)

    p = 1 - alpha (example 0.01 for 99%).
    LR_uc ~ Chi2(1)
    """
    T = len(hits)
    X = int(hits.sum())
    p = 1 - alpha
    pi_hat = X / T if T > 0 else 0.0

    # Careful for log(0)
    eps = 1e-12
    pi_hat = min(max(pi_hat, eps), 1 - eps)
    p = min(max(p, eps), 1 - eps)

    LR_uc = -2 * (
        (T - X) * np.log((1 - p) / (1 - pi_hat)) + X * np.log(p / pi_hat)
    )
    pval = 1 - chi2.cdf(LR_uc, df=1)
    return float(LR_uc), float(pval), X