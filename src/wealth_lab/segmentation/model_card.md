# Model Card — Client Segmentation

## Purpose
Create client segments to support tailored service, communications, and suitability discussions.

## Inputs
- Client/household features (here: synthetic, publish-safe)
- Optionally: derived portfolio behaviour features (volatility experienced, cashflow patterns)

## Method
KMeans clustering on standardised features.

## Outputs
- Segment labels per client
- Segment profiles (medians, counts)

## Limitations
- Segments are not "truth"; they are a lens for engagement.
- Use caution with sensitive attributes; consider fairness and explainability.
