# Model Card — Attribution & Factor Exposure

## Purpose
Explain portfolio performance:
- Brinson attribution vs a benchmark
- Rolling factor exposures (betas) and alpha proxies

## Inputs
- Portfolio and benchmark weights and returns by sleeve/asset
- Factor returns (e.g., Fama-French) and portfolio excess returns

## Outputs
- Attribution time series (allocation/selection/interaction)
- Rolling betas and alpha

## Limitations
- Factor model choice matters; alpha is not persistent evidence of skill.
- Brinson assumes clean sleeve mapping and consistent benchmarks.
