from __future__ import annotations
import pandas as pd

def fetch_sonia_monthly(start: str = "2005-01-01") -> pd.Series:
    """Fetch SONIA (when available) from FRED via pandas-datareader.
    If unavailable in your environment, this function raises; caller should fallback.
    """
    from pandas_datareader import data as pdr
    # Commonly referenced SONIA series on FRED (may change over time).
    # We'll attempt a couple candidate codes.
    candidates = ["IUDSOIA", "SONIA"]
    last_err = None
    for code in candidates:
        try:
            s = pdr.DataReader(code, "fred", start=start)[code]
            s = s.dropna()
            # Convert percent rate to monthly return approximation
            s_m = s.resample("M").last() / 100.0
            return s_m.rename("sonia_rate")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Unable to fetch SONIA from FRED. Last error: {last_err}")
