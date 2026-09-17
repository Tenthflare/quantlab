from abc import ABC, abstractmethod

import pandas as pd

from quantlab.data.prices import PITView


class Feature(ABC):
    name: str
    min_history: int = 0  # trading days of history required before the signal is valid

    @abstractmethod
    def compute(self, view: PITView, universe: list[str]) -> pd.Series:
        """
        Feature value per ticker as of `date_position`, using ONLY data dated <= date_position.
        Returns a Series indexed by ticker; a name not computable -> NaN."""
        ...
