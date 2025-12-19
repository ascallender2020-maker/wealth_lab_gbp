import pandas as pd
import numpy as np
from wealth_lab.attribution.brinson import brinson

def test_brinson_sums_to_active():
    idx = pd.date_range("2020-01-31", periods=3, freq="M")
    wp = pd.DataFrame([[0.6,0.4],[0.5,0.5],[0.7,0.3]], index=idx, columns=["E","B"])
    wb = pd.DataFrame([[0.5,0.5],[0.5,0.5],[0.5,0.5]], index=idx, columns=["E","B"])
    rp = pd.DataFrame([[0.02,0.01],[0.01,0.01],[0.03,0.0]], index=idx, columns=["E","B"])
    rb = pd.DataFrame([[0.015,0.008],[0.012,0.009],[0.02,0.005]], index=idx, columns=["E","B"])
    out = brinson(wp, wb, rp, rb)
    assert np.allclose(out["active"].values, (out["allocation"]+out["selection"]+out["interaction"]).values)
