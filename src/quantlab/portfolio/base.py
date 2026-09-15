from abc import ABC, abstractmethod

import pandas as pd


class PortfolioConstructor(ABC):
    @abstractmethod
    def weights(self, scores: pd.Series, prev_weights: pd.Series | None = None) -> pd.Series:
        """
        Map cross-sectional scores (per ticker, as of t) to target weights.
        Returns a Series over the universe; names with no position get 0.0."""
        ...
