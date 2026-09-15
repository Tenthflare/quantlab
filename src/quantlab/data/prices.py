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
        self.dates = panel.index.get_level_values("date").unique().sort_values()
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
        return dates

    def date_position(self, date: pd.Timestamp):
        """
        Calendar index of the last trading day <= date (-1 if before all data).
        """
        return int(self.dates.searchsorted(date, side="right")) - 1

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
            realised_return = realised_return.reindex(tickers)
        return realised_return
