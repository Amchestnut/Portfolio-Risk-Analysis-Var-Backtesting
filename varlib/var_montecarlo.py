import numpy as np
import pandas as pd

def rolling_montecarlo_var_es(
    return_dataframe: pd.DataFrame,
    weights: dict[str, float],
    alpha: float = 0.95,
    window: int = 250,
    n_simulations: int = 20000,
    lagged: bool = True,
    random_seed: int = 42,
) -> pd.Series:
    """
    For every day `t` use the window [t-window, t-1] to find μ (vector) and Σ,
    then simulate porfolio's 1-day return r_p ~ N(w'μ, w'Σw) using multivariate normal + Cholesky

    Returns rolling VaR.
    (ES can be calculated later, analogously)
    """

    rng = np.random.default_rng(random_seed)
    cols = list(return_dataframe.columns)
    W = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
    W = W / (np.sum(W) if np.sum(W) != 0 else 1.0)

    var_vals = []
    idx = return_dataframe.index

    for t in range(len(idx)):
        if t < window:
            var_vals.append(np.nan)
            continue
        # window_data = return_dataframe.iloc[t - window: t]
        window_data = return_dataframe.iloc[t - window: t].dropna(how="any")
        if len(window_data) < 2:
            var_vals.append(np.nan)
            continue

        mean = window_data.mean().to_numpy()
        # Covariance is N x N matrix, measures the "connection/correlation" between 2 instruments
        # On diagonals: there are variance of each instrument σ²
        # On all other fields: covariance of pairs (how instruments go with each other)
        cov = np.cov(window_data.to_numpy().T, ddof=1)

        # If Σ is simetric and positive, there is L so that Σ = L * Lᵀ
        # L is the Cholesky matrix of covariance (N x N)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # fallback: diagonal regularization
            cov = cov + 1e-8 * np.eye(cov.shape[0])
            L = np.linalg.cholesky(cov)

        # Matrix of independent standard normal numbers of Z dimension
        # rows: n - number of instruments
        # cols: number of simulations
        z = rng.standard_normal((window_data.shape[1], n_simulations))

        # L @ Z is the "CORRELATED" sum of return simulations
        correlated = L @ z  # shape (n_assets, n_sims)

        # Adding mean to every instruments
        # mean[:, None] is used to make (3,) into (3,1)
        shifted = mean[:, None] + correlated  # shape (n_assets, n_sims)

        sims = shifted.T  # shape (n_sims, n_assets)

        portfolio_sims = sims @ W  # portfolio simulated returns
        losses = -portfolio_sims
        var = np.quantile(losses, alpha)
        var_vals.append(float(var))

    var_series = pd.Series(var_vals, index=idx)

    if lagged:
        var_series = var_series.shift(1)
    return var_series.rename("VaR")