from __future__ import annotations
import pandas as pd

def brinson(wp: pd.DataFrame, wb: pd.DataFrame, rp: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic Brinson attribution: allocation, selection, interaction."""
    alloc = ((wp - wb) * rb).sum(axis=1)
    sel   = (wb * (rp - rb)).sum(axis=1)
    inter = ((wp - wb) * (rp - rb)).sum(axis=1)
    out = pd.DataFrame({"allocation": alloc, "selection": sel, "interaction": inter})
    out["active"] = out.sum(axis=1)
    return out
