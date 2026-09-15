from abc import ABC, abstractmethod

import pandas as pd


class CostModel(ABC):
    @abstractmethod
    def cost(self, trades: pd.Series) -> float:
        """
        Trading cost (in return units) for a vector of weight changes.
        """
        ...


class FixedBPSCost(CostModel):
    """
    Linear cost: `bps` basis points charged per unit of weight traded.
    """

    def __init__(self, bps: float = 10.0) -> None:
        self.rate = bps / 1e4  # 10 bps -> 0.001

    def cost(self, trades: pd.Series) -> float:
        return self.rate * float(trades.abs().sum())  # rate x turnover
