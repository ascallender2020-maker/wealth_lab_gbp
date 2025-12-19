from __future__ import annotations
import numpy as np
import pandas as pd

def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_ = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(X_, y, rcond=None)[0]
    return b  # [alpha, betas...]

def rolling_factor_exposure(
    factors: pd.DataFrame,
    excess_returns: pd.Series,
    window: int = 36,
) -> tuple[pd.Series, pd.DataFrame]:
    alphas = []
    betas = []
    idx = []
    for i in range(window, len(excess_returns) + 1):
        y = excess_returns.iloc[i-window:i].values
        X = factors.iloc[i-window:i].values
        b = ols_beta(X, y)
        idx.append(excess_returns.index[i-1])
        alphas.append(b[0])
        betas.append(b[1:])
    betas_df = pd.DataFrame(betas, index=idx, columns=factors.columns)
    alpha_s = pd.Series(alphas, index=idx, name="alpha")
    return alpha_s, betas_df
