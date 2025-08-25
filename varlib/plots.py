from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_var_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    """Return a VaR Series from either a Series or a DataFrame with column 'VaR'."""
    if isinstance(obj, pd.DataFrame):
        if "VaR" not in obj.columns:
            raise ValueError("DataFrame must contain a 'VaR' column.")
        return obj["VaR"]
    return obj


def plot_pnl_vs_var(
    port_ret: pd.Series,
    var_dict: Dict[str, pd.Series | pd.DataFrame],
    alpha: float,
    highlight: str = "HS",
    savepath: Optional[Path] = None,
) -> None:
    """
    Simple overlay: portfolio daily returns and -VaR lines for each method.
    We also mark exceedances for the 'highlight' method.
    """
    # Prepare data
    aligned = pd.concat(
        [port_ret.rename("r")] + [(-_to_var_series(v)).rename(f"-VaR {k}") for k, v in var_dict.items()],
        axis=1,
    ).dropna()

    if aligned.empty:
        raise ValueError("No overlap between returns and VaR series.")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(aligned.index, aligned["r"], linewidth=1.0, label="Portfolio return (r_t)")

    for k in var_dict.keys():
        ax.plot(aligned.index, aligned[f"-VaR {k}"], linewidth=1.0, label=f"-VaR {k}")

    # Exceedances for highlighted method: -r_t > VaR_t  <=> r_t < -VaR_t
    hl_col = f"-VaR {highlight}"
    hits = aligned["r"] < aligned[hl_col]
    ax.scatter(aligned.index[hits], aligned.loc[hits, "r"], s=18, marker="o", label=f"Exceedances ({highlight})")

    ax.set_title(f"P&L vs -VaR (alpha={alpha:.2f})")
    ax.set_ylabel("Daily return")
    ax.grid(True, linewidth=0.4, alpha=0.6)
    ax.legend(loc="best", ncol=2)

    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, bbox_inches="tight", dpi=140)
        plt.close(fig)
    else:
        plt.show()


def plot_kupiec_expected_vs_actual(
    backtest_table: pd.DataFrame,
    alpha: float,
    savepath: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing expected vs actual exceedances (Kupiec coverage).
    Assumes backtest_table contains columns: 'T' and 'exceedances' and index = method names.
    """
    if backtest_table.empty:
        raise ValueError("Backtest table is empty.")

    expected = (1 - alpha) * backtest_table["T"]
    actual = backtest_table["exceedances"]

    x = np.arange(len(backtest_table.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, expected.values, width, label="Expected")
    ax.bar(x + width / 2, actual.values, width, label="Actual")

    ax.set_xticks(x, backtest_table.index, rotation=20, ha="right")
    ax.set_title(f"Kupiec coverage: expected vs actual (alpha={alpha:.2f})")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.6)
    ax.legend(loc="best")

    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, bbox_inches="tight", dpi=140)
        plt.close(fig)
    else:
        plt.show()


def plot_mc_loss_histogram(
    ret_df: pd.DataFrame,
    weights: dict[str, float],
    window: int,
    n_sims: int,
    alpha: float,
    savepath: Optional[Path] = None,
) -> None:
    """
    Simple MC histogram for the last available day:
    - Calibrate mean and covariance on the trailing `window`.
    - Simulate 1-day portfolio returns.
    - Plot histogram of losses (-returns).
    """
    if len(ret_df) < window + 1:
        raise ValueError("Not enough data to calibrate the MC window.")

    # Use the last window [T-window, T-1]
    calib = ret_df.iloc[-window:]
    cols = list(calib.columns)
    W = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
    s = W.sum()
    if s == 0:
        raise ValueError("Weights sum to zero.")
    W = W / s

    mu = calib.mean().to_numpy()
    cov = np.cov(calib.to_numpy().T, ddof=1)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-8 * np.eye(cov.shape[0])
        L = np.linalg.cholesky(cov)

    rng = np.random.default_rng(42)
    z = rng.standard_normal((calib.shape[1], n_sims))
    sims = (mu[:, None] + L @ z).T
    port_sims = sims @ W
    losses = -port_sims

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(losses, bins=60)
    ax.set_title(f"Monte Carlo 1-day losses (window={window}, sims={n_sims}, alpha={alpha:.2f})")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Frequency")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, bbox_inches="tight", dpi=140)
        plt.close(fig)
    else:
        plt.show()
