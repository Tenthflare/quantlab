import json
import subprocess
from pathlib import Path

from quantlab.backtest.cost import FixedBPSCost
from quantlab.backtest.engine import BacktestEngine, month_end_rebalance_dates
from quantlab.data.fetch import fetch_clean_data, fetch_clean_data_sql, fetch_data_yf
from quantlab.data.prices import PriceStore
from quantlab.data.universe import dow_30
from quantlab.evaluation.metrics import Metrics
from quantlab.evaluation.tearsheet import tearsheet
from quantlab.evaluation.validation import permutation_pnl
from quantlab.features.momentum import Momentum
from quantlab.pipeline.load_config import RunConfig
from quantlab.portfolio.rank_dollar_neutral import RankDollarNeutral

UNIVERSES = {"DOW30": dow_30}
LOADERS = {"parquet": fetch_clean_data, "sql": fetch_clean_data_sql}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run_from_config(cfg: RunConfig) -> dict:
    universe = UNIVERSES[cfg.universe]()  # (tickers, name)
    tickers, _ = universe

    fetch_data_yf(universe, cfg.start_date)
    panel = LOADERS[cfg.storage](universe)
    store = PriceStore(panel)

    rebalance_dates = list(month_end_rebalance_dates(store.dates))
    engine = BacktestEngine(
        store,
        Momentum(cfg.lookback, cfg.lookback_end),
        RankDollarNeutral(cfg.k),
        FixedBPSCost(cfg.cost_bps),
    )
    result = engine.run(rebalance_dates, tickers)
    fwd_returns, turnover = result.records["pnl"], result.records["turnover"]
    periods_per_year = 12
    metrics = Metrics(fwd_returns, turnover, periods_per_year)
    stats = metrics.summary()
    perm = permutation_pnl(
        result,
        store,
        tickers,
        n_permutations=cfg.n_permutation,
        seed=cfg.seed,
    )

    out = Path(cfg.output_dir) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    tearsheet(
        result.records,
        title=f"QuantLab — {cfg.run_name}",
        permutation=perm,
        outpath=out / "tearsheet.png",
    )
    result.records.to_csv(out / "records.csv")
    (out / "metrics.json").write_text(
        json.dumps(
            {"stats": stats, "permutation": perm, "config": cfg.__dict__, "git_sha": _git_sha()},
            indent=2,
            default=str,
        )
    )
    return {"stats": stats, "permutation": perm}
