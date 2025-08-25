import numpy as np
import pandas as pd
from scipy.stats import norm


def emwa_vol(returns: pd.Series, lam: float = 0.94, min_periods: int = 30) -> pd.Series:
    """
    RiscMetrics EWMA variant for volatility (σ_t)
    σ_t^2 = (1-λ) * r_{t-1}^2 + λ * σ_{t-1}^2;
    lam = 0.94, its the daily default -> effective memory ~ 1/(1-λ) = 16, 17 days -> which means fast reaction for fresh volatility
    min_periods = first X days are NaN (we dont believe the prediction with little data)

    this is similar as "moving average", but with exponential weights, not equal.
    """

    var = (
        returns.pow(2)
        .ewm(alpha= 1-lam, adjust=False, min_periods=min_periods)
        .mean()
    )

    sig = np.sqrt(var)
    # sig[returns.expanding().count < min_periods] = np.nan

    return sig


def rolling_parametric_var_es(
    portfolio_returns: pd.Series,
    alpha: float = 0.95,
    window: int = 250,
    distribution: str = "normal",
    use_ewma: bool = False,
    ewma_lambda: float = 0.94,
    lagged: bool = True,
) -> pd.DataFrame:
    """
    Normal Var-Covar (student-t can add later).
    Loss L ~ N(μ_L, σ_L) sa μ_L = -μ_r, σ_L = σ_r.
    VaR_α = μ_L + z_α σ_L ; z_α = norm.ppf(α).
    ES_α = μ_L + σ_L * φ(z_α)/(1-α).

    mu (μ) = mean
    σ is EMWA (fast tracks changes, especially 2008 or 2020) or classical std (smoother, but slower reaction)
    ddof=1: unbiased assessment(procena)
    """
    if use_ewma:
        mean = portfolio_returns.rolling(window=window).mean()
        sigma = emwa_vol(portfolio_returns, lam=ewma_lambda)
    else:
        mean = portfolio_returns.rolling(window=window).mean()
        sigma = portfolio_returns.rolling(window=window).std(ddof=1)

    # (α-quantile of standard normal, 0.99 -> 2.3263, 0.95 -> 1.6449)
    z = norm.ppf(alpha)
    # loss convention, positive means loss.
    mean_L = -mean
    sigma_L = sigma

    # VaR & ES formulas (normal distribution)
    var_series = mean_L + z * sigma_L
    es_series = mean_L + (sigma_L * norm.pdf(z) / (1 - alpha))

    out = pd.DataFrame({"VaR": var_series, "ES": es_series})
    if lagged:
        out = out.shift(1)
    return out












