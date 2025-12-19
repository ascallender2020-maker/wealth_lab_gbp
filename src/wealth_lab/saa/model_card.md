# Model Card — Strategic Asset Allocation (SAA) Optimiser

## Purpose
Monthly multi-asset allocation for a GBP-base portfolio, balancing expected return, risk, and turnover.

## Inputs
- Monthly asset returns (ETF proxies)
- Rolling estimates of expected returns and covariance (default 60 months)

## Method
Convex optimisation:

maximize:  μᵀw  − λ·wᵀΣw − τ·||w − w_prev||₁  
subject to: sum(w)=1 and w_min ≤ w ≤ w_max

## Outputs
- Target weights per rebalance date
- Backtest NAV and turnover

## Assumptions / Limitations
- Expected returns estimated from history (high estimation error).
- Transaction costs are simplified (bps × turnover).
- Not investment advice; educational prototype.

## Monitoring ideas
- Weight stability & turnover
- Drift vs constraints
- Realised vs expected volatility
