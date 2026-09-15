# tests/data/test_prices.py
import pandas as pd
import pytest

from quantlab.data.prices import PriceStore


@pytest.fixture
def store() -> PriceStore:
    """
    Tiny hand-built panel. Two tickers, four trading days (Fri 01-03 → Mon 01-06
    skips the weekend, so we can test a non-trading `end`). Prices chosen so every
    return is trivial to check by eye."""
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    close = [100.0, 50.0, 105.0, 55.0, 105.0, 44.0, 210.0, 44.0]
    panel = pd.DataFrame(
        {"close_adj": close, "close_raw": close, "volume": [1] * 8},
        index=idx,
    ).sort_index()
    panel["daily_return"] = panel.groupby("ticker")["close_adj"].pct_change()
    return PriceStore(panel)


def test_history_never_returns_the_future(store):
    end_date = "2020-01-06"
    window = store.history_lookback(end_date, lookback=252)
    assert window.index.get_level_values("date").max() <= pd.Timestamp(end_date)


def test_history_boundary_on_a_non_trading_day(store):
    # Saturday 2020-01-04 is not a trading day → last row must be Fri 2020-01-03,
    # NOT Monday. This is the searchsorted(side="right") behaviour, asserted.
    window = store.history_lookback("2020-01-04", lookback=252)
    assert window.index.get_level_values("date").max() == pd.Timestamp("2020-01-03")


def test_history_lookback_counts_trading_days(store):
    # lookback=2 ending 01-07 → exactly the last two TRADING days: 01-06, 01-07.
    window = store.history_lookback("2020-01-07", lookback=2)
    dates_ = list(window.index.get_level_values("date").unique())
    assert dates_ == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")]


# ------------------------------------------------------------------------------
def test_realized_return_matches_hand_computed(store):
    r = store.calculate_realized_return("2020-01-02", "2020-01-03", tickers=["A", "B"])
    assert r["A"] == pytest.approx(0.05)  # 105/100 - 1
    assert r["B"] == pytest.approx(0.10)  #  55/50 - 1


def test_realized_return_reindexes_missing_ticker(store):
    # A name absent from the panel comes back NaN, not a KeyError.
    r = store.calculate_realized_return("2020-01-02", "2020-01-03", tickers=["A", "C"])
    assert r["A"] == pytest.approx(0.05)
    assert pd.isna(r["C"])


def test_price_at_returns_cross_section(store):
    p = store.price_at("2020-01-06", "close_adj")
    assert p["A"] == 105.0 and p["B"] == 44.0
