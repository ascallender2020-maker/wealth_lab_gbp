# Model Card — Tax-aware Rebalancing

## Purpose
Generate a monthly rebalance trade list that targets weights while minimising estimated realised capital gains.

## Inputs
- Current holdings (units)
- Current prices
- Target weights (from SAA)
- Tax lots (qty, cost basis, acquired date)

## Method
1) Compute desired unit trades to reach target weights
2) For sells, select lots in a gain-minimising order:
   - harvest losses first
   - then smallest gains

## Outputs
- Trade list (units)
- Estimated tax impact (demo-level CGT model)

## Limitations
- Jurisdiction-specific rules (e.g., UK share matching, bed-and-breakfasting, allowances) not modelled.
- This is a prototype; extend with accurate UK tax logic if needed.
