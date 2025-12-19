from __future__ import annotations
import pandas as pd

def fetch_fama_french_dev_factors() -> pd.DataFrame:
    """Fetch Fama-French Developed factors via pandas-datareader.
    Returns monthly factor returns in decimal form.
    """
    from pandas_datareader import data as pdr
    # This dataset name is supported by pandas-datareader; if it fails, caller should fallback.
    ds = pdr.DataReader("F-F_Research_Data_Factors", "famafrench")[0].copy()
    ds.index = pd.to_datetime(ds.index.astype(str))  # YYYYMM
    ds = ds / 100.0
    ds = ds.rename(columns={"Mkt-RF": "MKT_RF", "SMB": "SMB", "HML": "HML", "RF": "RF"})
    return ds[["MKT_RF", "SMB", "HML", "RF"]]
