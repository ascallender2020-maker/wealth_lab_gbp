from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import Universe, BacktestConfig, DEFAULT_UNIVERSE, DEFAULT_BT
from .data.prices_yf import fetch_adj_close, to_monthly_returns
from .saa.backtest import run_monthly_backtest
from .tax_rebal.tax_aware import (
    TaxConfig,
    compute_rebalance_trades,
    demo_make_random_lots,
    choose_lots_to_sell,
    estimate_tax_liability,
)
from .segmentation.synth_clients import make_synthetic_clients
from .segmentation.model import fit_segments, profile_segments
from .attribution.brinson import brinson
from .data.ff_factors import fetch_fama_french_dev_factors
from .attribution.factor import rolling_factor_exposure


ARTIFACTS_DIR = Path("artifacts")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_monthly_returns(universe: Universe = DEFAULT_UNIVERSE, start: str = "2008-01-01") -> pd.DataFrame:
    prices = fetch_adj_close(list(universe.tickers), start=start)
    rets = to_monthly_returns(prices)
    # Keep only fully populated rows for a clean demo
    return rets.dropna(how="any")


def build_saa_outputs(
    returns: pd.DataFrame,
    bt: BacktestConfig = DEFAULT_BT,
) -> pd.DataFrame:
    out = run_monthly_backtest(
        returns=returns,
        lookback_months=bt.lookback_months,
        transaction_cost_bps=bt.transaction_cost_bps,
    )
    return out


def build_tax_demo(
    prices_last: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float = 100_000.0,
    tax: TaxConfig | None = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """Create a demo tax-aware rebalance using simulated lots.

    Returns:
      - trades dataframe
      - estimated tax liability
      - lots sold table (for explanation)
    """
    if tax is None:
        tax = TaxConfig()

    prices = prices_last.to_dict()
    targets = target_weights.to_dict()

    # Create a slightly drifted current portfolio by perturbing weights
    rng = np.random.default_rng(seed)
    w0 = target_weights.to_numpy(dtype=float)
    w0 = np.clip(w0 + rng.normal(0.0, 0.02, size=len(w0)), 0.0, None)
    w0 = w0 / w0.sum()

    current_values = pd.Series(w0, index=target_weights.index) * float(portfolio_value)
    current_units = (current_values / prices_last).to_dict()

    lots = demo_make_random_lots(current_units, prices, seed=seed)

    # Compute trades to move toward targets
    trades = compute_rebalance_trades(
        holdings_units=current_units,
        prices=prices,
        target_weights=targets,
        portfolio_value=float(portfolio_value),
        tolerance=0.002,
    )
    # Convert to sell quantities by asset for lot selection
    qty_to_sell = {a: float(q) for a, q in trades[trades["side"] == "SELL"].set_index("asset")["qty"].to_dict().items()}
    sells = choose_lots_to_sell(lots, prices, qty_to_sell) if qty_to_sell else []

    tax_liab = estimate_tax_liability(sells, tax)

    sold_table = pd.DataFrame([{
        "asset": a,
        "sell_qty": q,
        "price": pxu,
        "cost_basis": lot.cost,
        "gain_per_unit": pxu - lot.cost,
        "acquired": lot.acquired.isoformat(),
    } for a, lot, q, pxu in sells])

    return trades, float(tax_liab), sold_table


def build_segmentation_outputs(n_clients: int = 800, k: int = 5, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clients = make_synthetic_clients(n_clients, seed=seed)
    _, labels = fit_segments(clients, k=k)
    profiles = profile_segments(clients, labels)
    labelled = clients.copy()
    labelled["segment"] = labels
    return labelled, profiles


def build_attribution_demo(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Simple attribution vs equal-weight benchmark using Brinson-style arithmetic.

    This is a demo: the "segments" are the assets in the universe.
    """
    rp = returns.loc[weights.index, weights.columns]
    wp = weights

    # Equal-weight benchmark
    wb = pd.DataFrame(
        np.full_like(wp.values, 1.0 / wp.shape[1], dtype=float),
        index=wp.index,
        columns=wp.columns,
    )
    rb = rp  # benchmark asset returns are the same market returns in this toy example

    return brinson(wp, wb, rp, rb)


def build_factor_outputs(
    portfolio_returns: pd.Series,
    window: int = 36,
) -> Tuple[pd.Series, pd.DataFrame]:
    ff = fetch_fama_french_dev_factors().copy()
    ff = ff.loc[portfolio_returns.index.min(): portfolio_returns.index.max()]
    ff = ff.dropna()

    aligned = portfolio_returns.loc[ff.index]
    excess = aligned - ff["RF"]
    factors = ff[["MKT_RF", "SMB", "HML"]]

    alpha, betas = rolling_factor_exposure(factors, excess, window=window)
    return alpha, betas


def save_artifacts(
    returns: pd.DataFrame,
    saa_bt: pd.DataFrame,
    trades: pd.DataFrame,
    tax_liab: float,
    sold_table: pd.DataFrame,
    clients: pd.DataFrame,
    profiles: pd.DataFrame,
    attribution: pd.DataFrame,
    alpha: pd.Series,
    betas: pd.DataFrame,
    universe: Universe = DEFAULT_UNIVERSE,
    bt_cfg: BacktestConfig = DEFAULT_BT,
    out_dir: Path = ARTIFACTS_DIR,
) -> None:
    ensure_dir(out_dir)

    returns.to_parquet(out_dir / "returns.parquet")
    saa_bt.to_parquet(out_dir / "saa_backtest.parquet")
    trades.to_parquet(out_dir / "tax_trades.parquet")
    sold_table.to_parquet(out_dir / "tax_sold_lots.parquet")
    pd.Series({"tax_liability": float(tax_liab)}).to_json(out_dir / "tax_summary.json")
    clients.to_parquet(out_dir / "clients.parquet")
    profiles.to_parquet(out_dir / "segment_profiles.parquet")
    attribution.to_parquet(out_dir / "attribution.parquet")
    alpha.to_frame().to_parquet(out_dir / "factor_alpha.parquet")
    betas.to_parquet(out_dir / "factor_betas.parquet")

    meta = {
        "universe": asdict(universe),
        "backtest": asdict(bt_cfg),
    }
    (out_dir / "meta.json").write_text(pd.Series(meta).to_json())


def load_artifacts(out_dir: Path = ARTIFACTS_DIR) -> Dict[str, object]:
    data: Dict[str, object] = {}
    data["returns"] = pd.read_parquet(out_dir / "returns.parquet")
    data["saa_backtest"] = pd.read_parquet(out_dir / "saa_backtest.parquet")
    data["tax_trades"] = pd.read_parquet(out_dir / "tax_trades.parquet")
    data["tax_sold_lots"] = pd.read_parquet(out_dir / "tax_sold_lots.parquet")
    data["tax_summary"] = pd.read_json(out_dir / "tax_summary.json", typ="series").to_dict()
    data["clients"] = pd.read_parquet(out_dir / "clients.parquet")
    data["segment_profiles"] = pd.read_parquet(out_dir / "segment_profiles.parquet")
    data["attribution"] = pd.read_parquet(out_dir / "attribution.parquet")
    data["factor_alpha"] = pd.read_parquet(out_dir / "factor_alpha.parquet")["alpha"]
    data["factor_betas"] = pd.read_parquet(out_dir / "factor_betas.parquet")
    return data


def build_all_and_save(
    universe: Universe = DEFAULT_UNIVERSE,
    bt_cfg: BacktestConfig = DEFAULT_BT,
    out_dir: Path = ARTIFACTS_DIR,
) -> None:
    returns = load_monthly_returns(universe)
    saa_bt = build_saa_outputs(returns, bt_cfg)

    # weights history from backtest
    w_cols = [c for c in saa_bt.columns if c.startswith("w_")]
    weights = saa_bt[w_cols].copy()
    weights.columns = [c[2:] for c in weights.columns]  # strip 'w_'

    # tax demo uses most recent month-end weights and prices
    prices = fetch_adj_close(list(universe.tickers), start=str(returns.index.min().date()))
    px_last = prices.resample("M").last().iloc[-1]
    w_last = weights.iloc[-1]

    trades, tax_liab, sold_table = build_tax_demo(px_last, w_last, seed=bt_cfg.seed)

    clients, profiles = build_segmentation_outputs(seed=bt_cfg.seed)
    attribution = build_attribution_demo(returns.loc[weights.index], weights)

    alpha, betas = build_factor_outputs(saa_bt["portfolio_return"], window=36)

    save_artifacts(
        returns=returns,
        saa_bt=saa_bt,
        trades=trades,
        tax_liab=tax_liab,
        sold_table=sold_table,
        clients=clients,
        profiles=profiles,
        attribution=attribution,
        alpha=alpha,
        betas=betas,
        universe=universe,
        bt_cfg=bt_cfg,
        out_dir=out_dir,
    )
