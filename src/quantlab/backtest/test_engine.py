import pandas as pd
import pytest
from quantlab.data.prices import PriceStore
from quantlab.portfolio.rank_dollar_neutral import RankDollarNeutral
from quantlab.backtest.cost import FixedBPSCost
from quantlab.backtest.engine import BacktestEngine

class FakeFeature:
    """Returns preset scores per date — decouples engine tests from momentum."""
    name = "fake"
    def __init__(self, by_date): self.by_date = by_date
    def compute(self, prices, date, universe):
        return self.by_date[pd.Timestamp(date)].reindex(universe)


@pytest.fixture
def store():
    dates = pd.to_datetime(["2020-01-02", "2020-02-03"])   # two rebalance dates
    tickers = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

    close = [100, 100, 100, 100,      110, 90, 105, 95]
    panel = pd.DataFrame({"close_adj": [float(x) for x in close],
                          "close_raw": [float(x) for x in close],
                          "volume": [1] * 8}, index=idx).sort_index()
    panel["daily_return"] = panel.groupby("ticker")["close_adj"].pct_change()
    return PriceStore(panel)


def test_engine_pnl_cost_and_turnover(store):
    t0 = pd.Timestamp("2020-01-02")
    # feature/score -> construct portfolio based on score -> run backtest
    feature = FakeFeature({t0: pd.Series({"A": 4, "B": 1, "C": 3, "D": 2})})
    engine = BacktestEngine(store, feature, RankDollarNeutral(k=1), FixedBPSCost(bps=10))
    result = engine.run(["2020-01-02", "2020-02-03"], ["A", "B", "C", "D"])

    # k=1: long A (+1), short B (-1). fwd_return: A=+0.10, B=-0.10.
    # gross = 1*0.10 + (-1)*(-0.10) = 0.20 ; turnover = 2 ; cost = 0.001*2 = 0.002
    row = result.records.loc[t0]
    assert row["gross_pnl"] == pytest.approx(0.20)
    assert row["turnover"]  == pytest.approx(2.0)
    assert row["cost"]      == pytest.approx(0.002)
    assert row["pnl"]       == pytest.approx(0.198)


def test_last_date_produces_no_record(store):
    feature = FakeFeature({pd.Timestamp("2020-01-02"): pd.Series({"A": 4, "B": 1, "C": 3, "D": 2})})
    engine = BacktestEngine(store, feature, RankDollarNeutral(k=1), FixedBPSCost(bps=10))
    result = engine.run(["2020-01-02", "2020-02-03"], ["A", "B", "C", "D"])
    assert len(result.records) == 1
    assert pd.Timestamp("2020-02-03") not in result.records.index