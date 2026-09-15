import numpy as np
import pandas as pd

from quantlab.backtest.engine import BacktestResult
from quantlab.data.prices import PriceStore


def permutation_pnl(
    records: BacktestResult,
    prices: PriceStore,
    universe: list[str],
    rebalance_dates: list[pd.Timestamp],
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Destroy the [signal -> forward-return] link by shuffling realized returns
    across ticker within each period, then recompute mean per-period PnL. Repeat
    many times to build a null distribution, and compare the real strategy to it.

    Because the book is dollar-neutral (weights sum to 0), the expected shuffled
    PnL is exactly 0 -- so the null centers at 0 by construction. A real edge shows
    up as real PnL sitting in the tail of that null (small p-value); a harness bug
    or leak shows up as a null that does NOT center at 0.
    """
    rng = np.random.default_rng(seed)
    dates = [pd.Timestamp(d) for d in rebalance_dates]
    pairs = list(zip(dates[:-1], dates[1:], strict=True))

    # Weights are already leak-free (computed from info <= t). Cache (w, fwd) per period.
    period_records = []
    for t, t_next in pairs:
        weight_t = records.weights.loc[t]
        fwd_return_t = prices.calculate_realized_return(t, t_next, universe).reindex(weight_t.index)
        period_records.append((weight_t.to_numpy(), fwd_return_t.to_numpy()))

    def mean_pnl(shuffle: bool) -> float:
        total = 0.0
        for weight, fwd_return in period_records:
            randomised_return = rng.permutation(fwd_return) if shuffle else fwd_return
            total += np.nansum(weight * randomised_return)
        return total / len(period_records)

    real = mean_pnl(shuffle=False)
    null = np.array([mean_pnl(shuffle=True) for _ in range(n_permutations)])
    p_value = (np.sum(np.abs(null) >= abs(real)) + 1) / (n_permutations + 1)  # +1: no p=0
    return {
        "real_mean_pnl": float(real),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "p_value": float(p_value),
    }
