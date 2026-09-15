import pandas as pd

from quantlab.data.prices import PriceStore


class ForwardReturn:
    """Return earned over the holding period [l, asof + horizon]. The ONLY
    forward-looking component in the system, and never fed back as a feature."""

    name = "forward_return"

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon  # holding period, trading days, equal to lookback_end

    def compute(self, prices: PriceStore, date: pd.Timestamp, universe: list[str]) -> pd.Series:
        date_idx = prices.date_position(date)
        date_idx_end = date_idx + self.horizon
        if date_idx < 0 or date_idx_end >= len(prices.dates):  # future window not fully available
            return pd.Series(index=pd.Index(universe, name="ticker"), dtype="float64")
        t_init, t_end = prices.dates[date_idx], prices.dates[date_idx_end]
        return prices.calculate_realized_return(t_init, t_end, universe)  # the forward read
