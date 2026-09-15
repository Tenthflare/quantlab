import pandas as pd

from quantlab.data.prices import PriceStore
from quantlab.features.base import Feature


class Momentum(Feature):
    """Cross-sectional momentum: cumulative close_adj return"""

    name = "momentum"

    def __init__(self, lookback: int, lookback_end: int) -> None:
        self.lookback = lookback  # start of window
        self.lookback_end = lookback_end  # end of window, excluding a period for trading itself

    def compute(self, prices: PriceStore, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        date_idx = prices.date_position(date)
        if date_idx < self.lookback:  # not enough history yet
            return pd.Series(index=pd.Index(universe, name="ticker"), dtype="float64")
        t_start = prices.dates[date_idx - self.lookback]
        t_end = prices.dates[date_idx - self.lookback_end]
        p_start = prices.price_at(t_start, "close_adj")
        p_end = prices.price_at(t_end, "close_adj")
        momentum = (p_end / p_start) - 1.0
        return momentum.reindex(universe)  # restrict to universe; missing -> NaN
