import pandas as pd
from pathlib import Path
from varlib.data_creator import load_prices_yf, to_log_returns
from varlib.plots import plot_pnl_vs_var, plot_kupiec_expected_vs_actual, plot_mc_loss_histogram
from varlib.returns import portfolio_returns
from varlib.var_history import history_var_expected_loss
from varlib.var_parametric import rolling_parametric_var_es
from varlib.var_montecarlo import rolling_montecarlo_var_es
from backtests.backtests import summarize_backtests

# 1) Config
tickers = ["AAPL", "MSFT", "AMZN", "TSM", "BA"]   # 5 liquid instruments
weights = {"AAPL": 0.25, "MSFT": 0.25, "AMZN": 0.25, "TSM": 0.15, "BA": 0.10}  # allocated capital in %
alpha_levels = [0.95, 0.99]     # probabilities for VaR
window = 365

# Prepare a place to save figures
fig_dir = Path("reports/figs")
fig_dir.mkdir(parents=True, exist_ok=True)

# 2) Data
prices = load_prices_yf(tickers, start="2022-01-01")
rets = to_log_returns(prices)

# This is everything in USD, if we mix with EUR, we need to convert.

# 3) Portfolio returns
r_p = portfolio_returns(rets, weights)

# 4) VaR models (rolling, out-of-sample)
results = {}

# Recompute (or reuse if you kept them) per-alpha VaR series for plotting
for a in alpha_levels:
    hs = history_var_expected_loss(r_p, alpha=a, window=window)
    par = rolling_parametric_var_es(r_p, alpha=a, window=window, use_ewma=False)
    par_ewma = rolling_parametric_var_es(r_p, alpha=a, window=window, use_ewma=True, ewma_lambda=0.94)
    mc = rolling_montecarlo_var_es(rets, weights, alpha=a, window=window, n_simulations=20000)

    # Build backtest table and store it
    tbl = pd.concat([
        summarize_backtests(r_p, hs["VaR"], a, f"HS ({int(a * 100)}%)"),
        summarize_backtests(r_p, par["VaR"], a, f"Parametric-N ({int(a * 100)}%)"),
        summarize_backtests(r_p, par_ewma["VaR"], a, f"Parametric-EWMA ({int(a * 100)}%)"),
        summarize_backtests(r_p, mc, a, f"MonteCarlo ({int(a * 100)}%)"),
    ])
    results[a] = tbl  # <- this prevents KeyError

    var_dict = {
        "HS": hs["VaR"],
        "Parametric-N": par["VaR"],
        "Parametric-EWMA": par_ewma["VaR"],
        "MonteCarlo": mc,
    }

    # 1) P&L vs -VaR (exceedances marked for HS)
    plot_pnl_vs_var(
        port_ret=r_p,
        var_dict=var_dict,
        alpha=a,
        highlight="HS",
        savepath=fig_dir / f"pnl_vs_var_alpha{int(a*100)}.png",
    )

    # 2) Kupiec expected vs actual (uses results[a] that is already build)
    plot_kupiec_expected_vs_actual(
        backtest_table=results[a],
        alpha=a,
        savepath=fig_dir / f"kupiec_expected_vs_actual_alpha{int(a*100)}.png",
    )

# 3) Simple Monte Carlo histogram (last day), pick one alpha (example: 0.99)
plot_mc_loss_histogram(
    ret_df=rets,
    weights=weights,
    window=window,
    n_sims=30000,
    alpha=0.99,
    savepath=fig_dir / "mc_loss_hist_alpha99.png",
)

print(f"\nSaved figures to: {fig_dir.resolve()}\n")


# 6) Summary
for a, tbl in results.items():
    print(f"\n=== Backtests @ alpha={a} ===")
    with pd.option_context('display.max_columns', None,
                           'display.width', None,
                           'display.max_colwidth', None):
        print(tbl.round(4))



