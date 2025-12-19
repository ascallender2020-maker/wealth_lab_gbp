from __future__ import annotations

import pandas as pd


def fetch_fama_french_dev_factors() -> pd.DataFrame:
    """Fetch Fama-French factors via pandas-datareader.

    Returns monthly factor returns in decimal form with columns:
      ["MKT_RF", "SMB", "HML", "RF"]

    Notes:
      - The raw dataset from pandas-datareader comes in percent units; we convert to decimals.
      - The index is monthly; we normalise to a Timestamp index at month-end for consistency.
    """
    from pandas_datareader import data as pdr

    # pandas-datareader returns a dict-like of tables; [0] is the main monthly factors table.
    ds = pdr.DataReader("F-F_Research_Data_Factors", "famafrench")[0].copy()

    # ds.index is typically an Int64Index like 192607, 192608, ... (YYYYMM). Convert robustly.
    idx = pd.Index(ds.index)

    # If it's already datetime-like, keep it; otherwise parse YYYYMM safely.
    if not isinstance(idx, pd.DatetimeIndex):
        # Ensure strings like "202407" so to_datetime can parse with format.
        s = idx.astype(str).str.zfill(6)
        dates = pd.to_datetime(s, format="%Y%m", errors="raise")
        # Put timestamps at month-end (common convention for monthly series)
        ds.index = dates.to_period("M").to_timestamp("M")
    else:
        # Normalise to month-end if it's datetime already
        ds.index = pd.to_datetime(ds.index).to_period("M").to_timestamp("M")

    # Convert percent -> decimal
    ds = ds / 100.0

    # Standardise column names
    rename_map = {
        "Mkt-RF": "MKT_RF",
        "SMB": "SMB",
        "HML": "HML",
        "RF": "RF",
    }
    ds = ds.rename(columns=rename_map)

    # Keep only what we need (and in a stable order)
    needed = ["MKT_RF", "SMB", "HML", "RF"]
    missing = [c for c in needed if c not in ds.columns]
    if missing:
        raise ValueError(f"Fama-French dataset missing columns {missing}. Got columns: {list(ds.columns)}")

    return ds[needed]
