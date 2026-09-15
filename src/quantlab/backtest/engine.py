from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    records: pd.DataFrame  # per-rebalance: gross_PnL, cost, PnL, turnover
    weights: pd.DataFrame  # date x ticker target weights


def month_end_rebalance_dates(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last actual trading day of each calendar month."""
    s = pd.Series(trading_dates, index=trading_dates)
    last = s.groupby([trading_dates.year, trading_dates.month]).last()
    return pd.DatetimeIndex(last.values)


class BacktestEngine:
    def __init__(self, prices, feature, portfolio, cost_model) -> None:
        self.prices = prices
        self.feature = feature
        self.portfolio = portfolio
        self.cost_model = cost_model

    def run(self, rebalance_dates: list[pd.Timestamp], universe: list[str]) -> BacktestResult:
        dates = [pd.Timestamp(d) for d in rebalance_dates]
        prev_weight = pd.Series(0.0, index=pd.Index(universe, name="ticker"))
        rows, weight_hist = [], {}

        # pairwise: decide at t, earn over (t -> t_next]. The LAST date has no
        # forward window, so it correctly produces no record.
        for t, t_next in zip(dates[:-1], dates[1:], strict=True):
            # compute features, execute trade, calculate return, repeat
            scores = self.feature.compute(self.prices, date=t, universe=universe)
            target_weight = self.portfolio.weights(scores, prev_weight)

            trades = target_weight.sub(prev_weight, fill_value=0.0)
            cost = self.cost_model.cost(trades)
            fwd_return = self.prices.calculate_realized_return(
                t, t_next, universe
            )  # earned AFTER t
            gross_return = float((target_weight * fwd_return).sum())
            # check for delisted stock or NaN
            held_asset = target_weight[target_weight != 0].index
            assert not fwd_return.reindex(held_asset).isna().any()

            rows.append(
                {
                    "date": t,
                    "gross_pnl": gross_return,
                    "cost": cost,
                    "pnl": gross_return - cost,
                    "turnover": float(trades.abs().sum()),
                }
            )
            weight_hist[t] = target_weight
            prev_weight = target_weight

        return BacktestResult(
            records=pd.DataFrame(rows).set_index("date"),
            weights=pd.DataFrame(weight_hist).T,
        )
