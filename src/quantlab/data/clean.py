import pandas as pd

from quantlab.data.cache import save_processed_data


def build_price_panel(universe_name: str, raw_df: pd.DataFrame) -> pd.DataFrame:
    panel = raw_df.stack(level="Ticker", future_stack=True)   # cols (field,ticker) -> rows (Date,Ticker)
    panel.index.names = ["Date", "Ticker"]
    panel = panel.sort_index()
    panel["Daily Return"] = panel.groupby("Ticker")["Adj Close"].pct_change()
    save_processed_data(universe_name, panel)
    return panel

