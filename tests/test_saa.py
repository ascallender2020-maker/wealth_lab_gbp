import numpy as np
import pandas as pd
from wealth_lab.saa.backtest import run_monthly_backtest

def test_backtest_runs():
    idx = pd.date_range("2015-01-31", periods=120, freq="M")
    rets = pd.DataFrame(np.random.default_rng(0).normal(0.005, 0.03, size=(120, 5)), index=idx, columns=list("ABCDE"))
    bt = run_monthly_backtest(rets, lookback_months=24)
    assert "portfolio_nav" in bt.columns
    assert bt.shape[0] == 120 - 24
