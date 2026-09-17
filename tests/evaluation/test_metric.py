import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.cost import FixedBPSCost
from quantlab.backtest.engine import BacktestEngine
from quantlab.data.prices import PriceStore
from quantlab.evaluation.metrics import Metrics
from quantlab.portfolio.rank_dollar_neutral import RankDollarNeutral


class FakeFeature:
    """Returns preset scores per date — decouples engine tests from momentum."""

    name = "fake"

    def __init__(self, by_date):
        self.by_date = by_date
        self.min_history: int = 0

    def compute(self, view, universe):
        return self.by_date[view.horizon].reindex(universe)


@pytest.fixture
def ran_backtest():
    rebalance_dates = [
        pd.Timestamp(d) for d in ["2020-01-31", "2020-02-28", "2020-03-31", "2020-04-30"]
    ]
    universe = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(42)
    prices = rng.uniform(100, 150, size=(len(rebalance_dates), len(universe)))
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
    results = engine.run(rebalance_dates, universe)
    return results, price_store, universe, rebalance_dates


def test_max_drawdown(ran_backtest):
    results, price_store, universe, rebalance_dates = ran_backtest
    fwd_returns, turnover = results.records["pnl"], results.records["turnover"]
    periods_per_year = 12
    metrics = Metrics(fwd_returns, turnover, periods_per_year)
    stats = metrics.summary()
    assert stats["max_drawdown"] >= -1.0


def test_max_drawdown_on_hand_computed_equity():
    pnl = pd.Series([0.10, -0.30])
    metrics = Metrics(pnl, pd.Series([1.0, 1.0]), 12)
    assert metrics.max_drawdown() == pytest.approx(-0.30, rel=1e-6)
