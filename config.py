from dataclasses import dataclass

@dataclass(frozen=True)
class Universe:
    # London tickers (GBP)
    tickers: tuple[str, ...] = ("IWDG.L", "IGLT.L", "SLXX.L", "INXG.L", "SGLN.L")
    names: tuple[str, ...] = (
        "Global Equity (GBP Hedged)",
        "UK Gilts",
        "GBP IG Credit",
        "UK Index-Linked Gilts",
        "Gold",
    )

@dataclass(frozen=True)
class BacktestConfig:
    rebalance_freq: str = "M"  # month-end
    lookback_months: int = 60
    transaction_cost_bps: float = 5.0
    seed: int = 42

DEFAULT_UNIVERSE = Universe()
DEFAULT_BT = BacktestConfig()
