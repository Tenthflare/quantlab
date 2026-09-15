import numpy as np
import pandas as pd


class Metrics:
    def __init__(self, fwd_return: pd.Series, turnover: pd.Series, periods_per_year: int):
        # records["pnl"]
        self.fwd_return = fwd_return
        self.turnover = turnover
        self.periods_per_year = periods_per_year
        self.num_datapoint = len(self.fwd_return)

    def cumulative_return(self) -> pd.Series:
        return (1.0 + self.fwd_return).cumprod()

    def annualised_return(self):
        if self.num_datapoint:
            return float(
                self.cumulative_return().iloc[-1] ** (self.periods_per_year / self.num_datapoint)
                - 1.0
            )
        else:
            float("nan")

    def annualised_volatility(self):
        return float(self.fwd_return.std(ddof=1) * np.sqrt(self.periods_per_year))

    def sharpe_ratio(self) -> float:
        sd = self.fwd_return.std(ddof=1)
        return (
            float("nan")
            if sd == 0
            else float(self.fwd_return.mean() / sd * np.sqrt(self.periods_per_year))
        )

    def sortino_ratio(self) -> float:
        """Like Sharpe, but penalizes only downside volatility."""
        neg_return = np.minimum(self.fwd_return, 0.0)  # 0 on up periods
        downside_deviation = np.sqrt(np.mean(np.square(neg_return)))
        if downside_deviation == 0:
            return float("nan")
        else:
            return float(
                (self.fwd_return.mean()) / downside_deviation * np.sqrt(self.periods_per_year)
            )

    def calmar_ratio(self) -> float:
        """Annualized return divided by the worst peak-to-trough drawdown."""
        ann_return = self.annualised_return()
        max_dd = abs(self.max_drawdown())
        return float("nan") if max_dd == 0 else float(ann_return / max_dd)

    def max_drawdown(self) -> float:
        return float((self.fwd_return / self.fwd_return.cummax() - 1.0).min())

    def hit_rate(self):
        return float((self.fwd_return > 0).mean())

    def avg_turnover(self):
        return float(self.turnover.mean())

    def summary(self) -> dict:
        return {
            "cumulative_return": self.cumulative_return(),
            "annualised_return": self.annualised_return(),
            "annualised_volatility": self.annualised_volatility(),
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "max_drawdown": self.max_drawdown(),
            "hit_rate": self.hit_rate(),
            "avg_turnover": self.avg_turnover(),
            "n_periods": self.num_datapoint,
        }
