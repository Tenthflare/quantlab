import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.cost import FixedBPSCost
from quantlab.backtest.engine import BacktestEngine
from quantlab.data.prices import PriceStore
from quantlab.evaluation.validation import permutation_pnl
from quantlab.portfolio.rank_dollar_neutral import RankDollarNeutral


class FakeFeature:
    """Returns preset scores per date — decouples engine tests from momentum."""

    name = "fake"

    def __init__(self, by_date):
        self.by_date = by_date

    def compute(self, prices, date, universe):
        return self.by_date[pd.Timestamp(date)].reindex(universe)


@pytest.fixture
def ran_backtest():
    rebalance_dates = [
        pd.Timestamp(d) for d in ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]
    ]
    universe = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    prices = rng.uniform(50, 150, size=(len(rebalance_dates), len(universe)))
    idx = pd.MultiIndex.from_product([rebalance_dates, universe], names=["date", "ticker"])
    panel = pd.DataFrame(
        {"close_adj": prices.ravel(), "close_raw": prices.ravel(), "volume": 1}, index=idx
    ).sort_index()
    panel["daily_return"] = panel.groupby("ticker")["close_adj"].pct_change()
    price_store = PriceStore(panel)
    scores = {d: pd.Series(rng.normal(size=len(universe)), index=universe) for d in rebalance_dates}
    engine = BacktestEngine(
        price_store, FakeFeature(scores), RankDollarNeutral(k=2), FixedBPSCost(0)
    )
    records = engine.run(rebalance_dates, universe)
    return records, price_store, universe, rebalance_dates


def test_shuffled_labels_center_at_zero(ran_backtest):
    records, price_store, universe, rebalance_dates = ran_backtest
    out = permutation_pnl(
        records, price_store, universe, rebalance_dates, n_permutations=3000, seed=1
    )
    # Dollar-neutral book => shuffled PnL has expectation 0. The Monte-Carlo mean
    # should be within a few standard errors of zero.
    standard_error = out["null_std"] / np.sqrt(3000)
    assert abs(out["null_mean"]) < 5 * standard_error
