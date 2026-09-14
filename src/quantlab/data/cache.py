from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

RAW = REPO_ROOT / "data" / "raw"
PROC = REPO_ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

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