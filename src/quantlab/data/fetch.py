import nasdaqdatalink
import pandas as pd
import yfinance as yf

from quantlab.data.cache import raw_exists, load_raw_data, save_raw_data, processed_exists, load_processed_data
from universe import dow_30

def fetch_data_sharadar(universe, start_date: str, force_refetch=False) -> pd.DataFrame:
    """force_refetch for re-download in case of data change/updates"""
    from quantlab.config import NASDAQ_API_KEY
    nasdaqdatalink.ApiConfig.api_key = NASDAQ_API_KEY
    ticker_list, universe_name = universe
    if raw_exists(universe_name) and not force_refetch:
        return load_raw_data(universe_name)
    else:
        raw_df = nasdaqdatalink.get_table(
               "SHARADAR/SEP",
               ticker=ticker_list,                 # all names in a single query
               date={"gte": start_date},   # bound the range to cut request volume
               paginate=True,
           )
        save_raw_data(universe_name, raw_df)
        return raw_df

def fetch_data_yf(universe, start_date: str, force_refetch=False) -> pd.DataFrame:
    ticker_list, universe_name = universe
    if raw_exists(universe_name) and not force_refetch:
        return load_raw_data(universe_name)
    else:
        raw_df = yf.download(ticker_list, start=start_date, auto_adjust=False)
        save_raw_data(universe_name, raw_df)
        return raw_df

def build_price_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    panel = raw_df.stack(level="Ticker", future_stack=True)   # cols (field,ticker) -> rows (Date,Ticker)
    panel.index.names = ["date", "ticker"]
    panel = panel.rename(columns={"Close": "close_raw", "Volume": "volume",
                                  "Adj Close": "close_adj"})
    panel = panel[["close_adj", "close_raw", "volume"]].sort_index()
    panel["daily_return"] = panel.groupby("ticker")["close_adj"].pct_change()

    return panel

def fetch_clean_data(universe) -> pd.DataFrame:
    ticker_list, universe_name = universe
    if processed_exists(universe_name):
        return load_processed_data(universe_name)
    else:
        raw_df = load_raw_data(universe_name)
        return build_price_panel(raw_df)

