from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import date
import numpy as np
from .lots import Lot

@dataclass
class TaxConfig:
    # For demo only: configurable CGT rate; you can model ISA/SIPP by setting cg_rate=0.
    cg_rate: float = 0.20
    loss_offset: bool = True

def gain_per_unit(lot: Lot, px: float) -> float:
    return px - lot.cost

def choose_lots_to_sell(
    lots: List[Lot],
    prices: Dict[str, float],
    qty_to_sell: Dict[str, float],
) -> List[Tuple[str, Lot, float, float]]:
    """Greedy tax-aware lot selection: losses first, then smallest gains."""
    sells: List[Tuple[str, Lot, float, float]] = []
    for asset, need in qty_to_sell.items():
        if need <= 0:
            continue
        px = float(prices[asset])
        candidates = [l for l in lots if l.asset == asset and l.qty > 0]
        candidates.sort(key=lambda l: gain_per_unit(l, px))  # most negative first

        remaining = float(need)
        for lot in candidates:
            if remaining <= 0:
                break
            q = min(lot.qty, remaining)
            sells.append((asset, lot, q, px))
            remaining -= q

        if remaining > 1e-9:
            raise ValueError(f"Insufficient quantity to sell for {asset}")
    return sells

def estimate_tax_liability(sells, tax: TaxConfig) -> float:
    realised = 0.0
    for _, lot, q, px in sells:
        realised += (px - lot.cost) * q
    taxable = realised
    if tax.loss_offset:
        taxable = max(0.0, taxable)
    return taxable * tax.cg_rate

def compute_rebalance_trades(
    holdings_units: Dict[str, float],
    prices: Dict[str, float],
    target_weights: Dict[str, float],
    nav: float,
) -> Dict[str, float]:
    """Compute desired trade quantities (+buy, -sell) in units."""
    trades = {}
    for asset, w_t in target_weights.items():
        px = float(prices[asset])
        target_value = nav * float(w_t)
        current_value = float(holdings_units.get(asset, 0.0)) * px
        diff_value = target_value - current_value
        trades[asset] = diff_value / px
    return trades

def demo_make_random_lots(assets: List[str], holdings_units: Dict[str, float], prices: Dict[str, float], seed: int = 42) -> List[Lot]:
    """Create synthetic lots consistent with current holdings, for demo purposes."""
    rng = np.random.default_rng(seed)
    lots: List[Lot] = []
    for a in assets:
        qty = float(holdings_units.get(a, 0.0))
        if qty <= 0:
            continue
        # split into 3 lots
        splits = rng.dirichlet([1, 1, 1])
        for k in range(3):
            q = qty * float(splits[k])
            px = float(prices[a])
            # random historical cost around current px
            cost = px * float(rng.uniform(0.7, 1.2))
            acquired = date(int(rng.integers(2016, 2024)), int(rng.integers(1, 13)), int(rng.integers(1, 28)))
            lots.append(Lot(asset=a, qty=q, cost=cost, acquired=acquired))
    return lots
