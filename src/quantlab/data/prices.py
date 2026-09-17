import pandas as pd


class PriceStore:
    """
    Point-in-time (PIT) price server. The system queries for prices for backtest/trading here.

    Format: DataFrame indexed by MultiIndex[date, ticker],
    sorted, with columns [close_adj, close_raw, volume, daily_return].
    """

    def __init__(self, panel: pd.DataFrame) -> None:
        panel = panel.sort_index()
        self.panel = panel
        self.dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique().sort_values())
        # start and end date may not be within the specified start_date and end_date
        # (works for yfinance format)
        self.start_date_trading = self.panel.index.get_level_values("date")[0]
        self.end_date_trading = self.panel.index.get_level_values("date")[-1]

    def trading_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Iterates through trading dates from start_date to end_date. Must specify both.
        """
        assert start_date >= self.start_date_trading and end_date <= self.end_date_trading
        selected_range = self.panel.loc[start_date:end_date]
        dates = selected_range.index.get_level_values("date").unique().sort_values()
        return pd.DatetimeIndex(dates)

    def date_position(self, date: pd.Timestamp):
        """
        Calendar index of the last trading day <= date (-1 if before all data).
        """
        date = pd.Timestamp(date)  # guardrail in case str is passed
        position = int(self.dates.searchsorted(date, side="right")) - 1
        assert position < 0 or self.dates[position] <= date
        return position

    def history_lookback(self, end_date: pd.Timestamp, lookback: int):
        """
        The last `lookback` TRADING days up to and including `end`.
        `lookback` is in trading days (rows of the calendar), not calendar days.
        """
        date_idx = self.date_position(end_date)
        if date_idx < 0:
            return self.panel.iloc[:0]
        lo = self.dates[max(0, date_idx - lookback + 1)]
        hi = self.dates[date_idx]  # last trading day <= end
        window = self.panel.loc[lo:hi]
        assert window.index.get_level_values("date").max() <= pd.Timestamp(
            end_date
        )  # firewall guard
        return window

    def price_at(self, date: pd.Timestamp, field="close_adj") -> pd.Series:
        """Cross-section of the universe at one trading date for a given field."""
        return self.panel.loc[date][field]

    def calculate_realized_return(
        self, from_date: pd.Timestamp, to_date: pd.Timestamp, tickers: str | list[str]
    ) -> pd.Series:
        """
        Return over (frm -> to].
        Uses close_adj (split+div adjusted).
        """
        price_init = self.price_at(from_date, "close_adj")
        price_final = self.price_at(to_date, "close_adj")
        realised_return = (price_final / price_init) - 1.0
        if tickers is not None:
            if isinstance(tickers, str):
                tickers = [tickers]
            realised_return = realised_return.reindex(tickers)
        return realised_return


class PITView:
    """
    Bounded point-in-time view. Features receive this, never the
    full store, so future access is impossible by construction.
    """

    def __init__(self, store: PriceStore, horizon) -> None:
        self._store = store
        self.horizon = pd.Timestamp(horizon)
        self.horizon_position = store.date_position(horizon)
        self._dates = self._store.dates[: self.horizon_position + 1]

    def price_at(self, date, field="close_adj") -> pd.Series:
        assert pd.Timestamp(date) <= self.horizon, "PIT violation: future access"
        return self._store.price_at(date, field)

    def history(self, lookback: int):
        return self._store.history_lookback(self.horizon, lookback)

    # NOTE: there is deliberately NO realized_return / forward method here.
