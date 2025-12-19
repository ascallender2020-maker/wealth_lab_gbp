from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


def _fallback_optimiser(
    w_prev: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    risk_aversion: float,
    turnover_aversion: float,
) -> np.ndarray:
    """Fallback optimiser when cvxpy isn't available.

    This is NOT a production optimiser; it's a deterministic heuristic:
    - compute unconstrained mean-variance direction: (cov + eps I)^-1 mu
    - blend with previous weights to reduce turnover
    - project to bounds + simplex
    """
    n = len(mu)
    eps = 1e-3
    cov_reg = cov + eps * np.eye(n)
    try:
        raw = np.linalg.solve(cov_reg, mu)
    except np.linalg.LinAlgError:
        raw = mu.copy()

    # make long-only preference
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 1e-12:
        raw = np.ones(n)

    w = raw / raw.sum()

    # turnover control: blend towards previous weights
    alpha = 1.0 / (1.0 + 10.0 * float(turnover_aversion))
    w = alpha * w + (1.0 - alpha) * w_prev

    # bounds + renormalise
    w = np.clip(w, w_min, w_max)
    if w.sum() <= 1e-12:
        w = np.clip(np.ones(n) / n, w_min, w_max)

    w = w / w.sum()
    return w


def optimise_saa(
    w_prev: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    risk_aversion: float = 5.0,
    turnover_aversion: float = 0.10,
) -> np.ndarray:
    """Monthly SAA optimiser.

    Objective (cvxpy path):
        maximise mu'w - λ w'Σw - τ ||w-w_prev||_1
    subject to:
        sum(w)=1 and bounds

    If cvxpy isn't installed, falls back to a simple deterministic heuristic.
    """
    if cp is None:
        return _fallback_optimiser(w_prev, mu, cov, w_min, w_max, risk_aversion, turnover_aversion)

    n = len(mu)
    w = cp.Variable(n)

    ret = mu @ w
    risk = cp.quad_form(w, cov)
    turnover = cp.norm1(w - w_prev)

    prob = cp.Problem(
        cp.Maximize(ret - risk_aversion * risk - turnover_aversion * turnover),
        [
            cp.sum(w) == 1,
            w >= w_min,
            w <= w_max,
        ],
    )
    prob.solve(solver=cp.OSQP, verbose=False)

    if w.value is None:
        # fall back rather than failing the whole run
        return _fallback_optimiser(w_prev, mu, cov, w_min, w_max, risk_aversion, turnover_aversion)

    wv = np.asarray(w.value, dtype=float)
    wv = np.clip(wv, w_min, w_max)
    if wv.sum() <= 1e-12:
        wv = np.clip(np.ones(n) / n, w_min, w_max)
    wv = wv / wv.sum()
    return wv
