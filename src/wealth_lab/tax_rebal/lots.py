from __future__ import annotations
from dataclasses import dataclass
from datetime import date

@dataclass
class Lot:
    asset: str
    qty: float
    cost: float      # cost per unit
    acquired: date
