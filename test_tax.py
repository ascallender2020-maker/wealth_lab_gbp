from datetime import date

from wealth_lab.tax_rebal.lots import Lot
from wealth_lab.tax_rebal.tax_aware import choose_lots_to_sell, estimate_tax_liability, TaxConfig


def test_choose_lots_prefers_losses_first():
    lots = [
        Lot("A", 10, 100, date(2020, 1, 1)),
        Lot("A", 10, 50, date(2021, 1, 1)),   # big gain if price=120
        Lot("A", 10, 140, date(2022, 1, 1)),  # loss if price=120
    ]
    sells = choose_lots_to_sell(lots, {"A": 120}, {"A": 10})
    # first sell should be loss lot (cost 140)
    assert sells[0][1].cost == 140


def test_tax_non_negative_with_loss_offset():
    lots = [Lot("A", 10, 140, date(2022, 1, 1))]
    sells = choose_lots_to_sell(lots, {"A": 120}, {"A": 10})
    tax = estimate_tax_liability(sells, TaxConfig(cg_rate=0.2, loss_offset=True))
    assert tax == 0.0


def test_tax_can_be_negative_if_loss_offset_disabled():
    lots = [Lot("A", 10, 140, date(2022, 1, 1))]
    sells = choose_lots_to_sell(lots, {"A": 120}, {"A": 10})
    tax = estimate_tax_liability(sells, TaxConfig(cg_rate=0.2, loss_offset=False))
    assert tax < 0.0
