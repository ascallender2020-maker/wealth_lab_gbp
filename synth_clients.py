from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic_clients(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Create synthetic client attributes (safe to publish)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 80, size=n)
    horizon_years = rng.integers(2, 30, size=n)
    income_need = rng.uniform(0, 1, size=n)
    risk_score = rng.normal(0, 1, size=n)
    contrib_monthly = rng.lognormal(mean=6.5, sigma=0.6, size=n)  # arbitrary scale
    drawdown_aversion = rng.uniform(0, 1, size=n)

    df = pd.DataFrame({
        "age": age,
        "horizon_years": horizon_years,
        "income_need": income_need,
        "risk_score": risk_score,
        "monthly_contrib": contrib_monthly,
        "drawdown_aversion": drawdown_aversion,
    })
    return df
