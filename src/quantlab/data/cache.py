from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parents[3]

RAW = REPO_ROOT / "data" / "raw"
PROC = REPO_ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# One SQLite file; one table per universe (table name = cache key, e.g. "DOW30").
_engine = create_engine(f"sqlite:///{PROC / 'quantlab.db'}")


def _create_table(name: str) -> None:
    """
    PK (ticker, date) + a date index.
    """
    with _engine.begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        conn.execute(
            text(f'''
            CREATE TABLE "{name}" (
                date         TEXT    NOT NULL,
                ticker       TEXT    NOT NULL,
                close_adj    REAL,
                close_raw    REAL,
                volume       INTEGER,
                daily_return REAL,
                PRIMARY KEY (ticker, date)
            )''')
        )
        conn.execute(text(f'CREATE INDEX "idx_{name}_date" ON "{name}"(date)'))


def save_processed_data_sql(name: str, df: pd.DataFrame) -> None:
    """
    Persist the canonical [date, ticker] panel to a SQLite table.
    """
    flat = df.reset_index()
    flat["date"] = pd.to_datetime(flat["date"]).dt.strftime("%Y-%m-%d")
    _create_table(name)
    flat.to_sql(name, _engine, if_exists="append", index=False)


def load_processed_data_sql(name: str) -> pd.DataFrame:
    """
    Read the table back into the canonical [date, ticker] panel.
    """
    df = pd.read_sql(text(f'SELECT * FROM "{name}"'), _engine, parse_dates=["date"])
    return df.set_index(["date", "ticker"]).sort_index()


def load_processed_pit(name: str, pit_date) -> pd.DataFrame:
    """
    Point-in-time (PIT) load: only rows dated <= pit_date, filtered in SQL.
    """
    q = text(f'SELECT * FROM "{name}" WHERE date <= :pit_date')
    df = pd.read_sql(
        q,
        _engine,
        params={"pit_date": pd.Timestamp(pit_date).strftime("%Y-%m-%d")},
        parse_dates=["date"],
    )
    return df.set_index(["date", "ticker"]).sort_index()


def processed_sql_exists(name: str) -> bool:
    return inspect(_engine).has_table(name)


# ===================================================================================
# Save and load raw/processed data from parquet
def save_raw_data(name: str, df: pd.DataFrame) -> None:
    """Cache raw dataframe to parquet"""
    df.to_parquet(RAW / f"{name}.parquet")


def load_raw_data(name: str) -> pd.DataFrame:
    """Load cached dataframe"""
    return pd.read_parquet(RAW / f"{name}.parquet")


def raw_exists(name: str) -> bool:
    """Check if cached data exists"""
    return (RAW / f"{name}.parquet").exists()


def save_processed_data(name: str, df: pd.DataFrame) -> None:
    """Cache raw dataframe to parquet"""
    df.to_parquet(PROC / f"{name}.parquet")


def load_processed_data(name: str) -> pd.DataFrame:
    """Load cached dataframe"""
    return pd.read_parquet(PROC / f"{name}.parquet")


def processed_exists(name: str) -> bool:
    """Check if cached data exists"""
    return (PROC / f"{name}.parquet").exists()
