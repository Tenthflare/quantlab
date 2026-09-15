import pandas as pd

from quantlab.portfolio.base import PortfolioConstructor


class RankDollarNeutral(PortfolioConstructor):
    """
    Dollar-neutral long-short. Equal-weight the top-k scores long and the
    bottom-k short: long leg sums to +1, short leg to -1 (net 0, gross 2).
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def weights(self, scores: pd.Series, prev_weights: pd.Series | None = None) -> pd.Series:
        valid_scores = scores.dropna().sort_values()  # ascending; NaN can't be ranked
        default_weight = pd.Series(0.0, index=scores.index)  # default: no position
        if len(valid_scores) < 2 * self.k:  # there must be at least 2 assets to long/short
            return default_weight
        shorts = valid_scores.index[: self.k]  # lowest scores
        longs = valid_scores.index[-self.k :]  # highest scores
        default_weight[longs] = 1.0 / self.k
        default_weight[shorts] = -1.0 / self.k
        return default_weight
