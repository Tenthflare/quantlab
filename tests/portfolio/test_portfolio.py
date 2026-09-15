import pandas as pd
import pytest

from quantlab.portfolio.rank_dollar_neutral import RankDollarNeutral


def test_dollar_neutral_equal_weight():
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0})
    w = RankDollarNeutral(k=2).weights(scores)
    assert w[["A", "B"]].tolist() == [0.5, 0.5]  # top-2 long
    assert w[["E", "F"]].tolist() == [-0.5, -0.5]  # bottom-2 short
    assert w[["C", "D"]].tolist() == [0.0, 0.0]  # middle flat
    assert w.sum() == pytest.approx(0.0)  # net 0 -> dollar-neutral
    assert w.abs().sum() == pytest.approx(2.0)  # gross 2x


def test_nan_scores_never_get_a_position():
    scores = pd.Series({"A": 5, "B": float("nan"), "C": 3, "D": 1})
    w = RankDollarNeutral(k=1).weights(scores)
    assert w["B"] == 0.0
    assert w["A"] == pytest.approx(1.0) and w["D"] == pytest.approx(-1.0)
    assert w.sum() == pytest.approx(0.0)


def test_flat_book_when_too_few_names():
    w = RankDollarNeutral(k=2).weights(pd.Series({"A": 5, "B": 1}))  # need 4, have 2
    assert (w == 0.0).all()
