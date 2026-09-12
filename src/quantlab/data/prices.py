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
        # start and end date may not be within the specified start_date and end_date (works for yfinance format)
        self.start_date_trading = self.panel.index.get_level_values("date")[0]
        self.end_date_trading = self.panel.index.get_level_values("date")[-1]

    def trading_dates(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """
        Iterates through trading dates from start_date to end_date. Must specify both.
        """
        assert start_date >= self.start_date_trading and end_date <= self.end_date_trading
        selected_range = self.panel.loc[start_date:end_date]
        dates = selected_range.index.get_level_values("date").unique().sort_values()
        return dates

    def price_at(self, date, field="close_adj") -> pd.Series:
        """Cross-section of the universe at one trading date for a given field."""
        return self.panel.loc[date][field]

    def calculate_realized_return(self, from_date: str,
                                  to_date: str,
                                  tickers: str | list[str]) -> pd.Series:
        """
        Return over (frm -> to].
        Uses close_adj (split+div adjusted).
        """
        price_init = self.price_at(from_date, "close_adj")
        price_final = self.price_at(to_date, "close_adj")
        realised_return = (price_final / price_init) - 1.0
        if tickers is not None:
            realised_return = realised_return[tickers]
        return realised_return

