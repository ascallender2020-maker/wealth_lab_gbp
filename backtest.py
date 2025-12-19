from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import optimise_saa


def estimate_mu_cov(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate expected returns and covariance from a returns window."""
    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float)
    return mu, cov


def run_monthly_backtest(
    returns: pd.DataFrame,
    lookback_months: int = 60,
    w_min: float = 0.0,
    w_max: float = 0.60,
    risk_aversion: float = 5.0,
    turnover_aversion: float = 0.10,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Run a simple monthly backtest for the SAA optimiser.

    Parameters
    ----------
    returns:
        Monthly returns (index = month-end dates, columns = assets).
    lookback_months:
        Rolling window length used to estimate mu/cov.
    w_min, w_max:
        Per-asset bounds (long-only default).
    risk_aversion:
        Lambda in mean-variance objective.
    turnover_aversion:
        Tau in L1 turnover penalty.
    transaction_cost_bps:
        Applied to turnover each rebalance (bps of traded notional).

    Returns
    -------
    DataFrame with portfolio returns, nav, turnover, and weights per asset.
    """
    if returns.isna().all(axis=None):
        raise ValueError("returns is all-NaN")
    returns = returns.dropna(how="any")  # keep simple and deterministic

    tickers = list(returns.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("need at least 2 assets")

    # Bounds vectors
    wmin = np.full(n, float(w_min))
    wmax = np.full(n, float(w_max))
    if (wmin < 0).any() or (wmax <= 0).any():
        raise ValueError("invalid bounds")
    if float(w_max) * n < 1 - 1e-9:
        raise ValueError("bounds infeasible: sum(w_max) < 1")

    # Start from equal weights
    w_prev = np.full(n, 1.0 / n, dtype=float)

    rows: list[dict] = []
    for i in range(lookback_months, len(returns)):
        end = returns.index[i]

        window = returns.iloc[i - lookback_months : i]
        mu, cov = estimate_mu_cov(window)

        w = optimise_saa(
            w_prev=w_prev,
            mu=mu,
            cov=cov,
            w_min=wmin,
            w_max=wmax,
            risk_aversion=risk_aversion,
            turnover_aversion=turnover_aversion,
        )

        turnover = float(np.abs(w - w_prev).sum())
        tc = (transaction_cost_bps / 10000.0) * turnover

        r_next = float(returns.iloc[i].to_numpy(dtype=float) @ w) - tc

        row = {
            "date": end,
            "portfolio_return": r_next,
            "turnover": turnover,
        }
        for j, t in enumerate(tickers):
            row[f"w_{t}"] = float(w[j])
        rows.append(row)

        w_prev = w

    out = pd.DataFrame(rows).set_index("date")
    out["portfolio_nav"] = (1.0 + out["portfolio_return"]).cumprod()
    return out
