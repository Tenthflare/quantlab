# tests/features/test_momentum.py
import pandas as pd
import pytest

from quantlab.data.prices import PITView, PriceStore
from quantlab.features.momentum import Momentum
from quantlab.labels.forward_return import ForwardReturn


@pytest.fixture
def store():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    close = [10.0, 11.0, 12.0, 20.0, 21.0]
    panel = pd.DataFrame(
        {"close_adj": close, "close_raw": close, "volume": [1] * 5}, index=idx
    ).sort_index()
    panel["daily_return"] = panel.groupby("ticker")["close_adj"].pct_change()
    return PriceStore(panel)


def test_momentum_uses_skip_window(store):
    # should not use the current date "2020-01-08"
    momentum = Momentum(lookback=3, lookback_end=1).compute(
        PITView(store, pd.Timestamp("2020-01-08")), ["A"]
    )
    assert momentum["A"] == pytest.approx(20.0 / 11.0 - 1.0)


def test_momentum_nan_when_history_too_short(store):
    momentum = Momentum(lookback=3, lookback_end=1).compute(
        PITView(store, pd.Timestamp("2020-01-06")), ["A"]
    )
    assert pd.isna(momentum["A"])  # idx 2 < lookback 3


def test_forward_return_is_earned_after_date_position(store):
    fwd_ret = ForwardReturn(horizon=1).compute(store, pd.Timestamp("2020-01-02"), ["A"])
    assert fwd_ret["A"] == pytest.approx(11.0 / 10.0 - 1.0)
