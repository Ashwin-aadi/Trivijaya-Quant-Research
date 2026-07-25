"""Rank on a momentum z-score computed across the universe on each date."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, stdev, top_n, window_return


class CrossSectionalZscore(Strategy):
    """Standardises monthly returns against that day's cross-section, then ranks."""

    rationale = (
        "Standardising against the cross-section on each date keeps the ranking comparable "
        "through calm and turbulent periods alike, since a five percent move means something "
        "different in each. Only contemporaneous data enters the calculation."
    )

    def __init__(self, lookback: int = 21, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        values = list(returns.values())
        mean = sum(values) / len(values)
        dispersion = stdev(values)
        if dispersion <= 0:
            return Signal(information_available_at=stamp, weights={})

        # The mean and dispersion come from this date's cross-section only — never a pooled
        # statistic over time, which would import information from other periods.
        scores = {sym: (r - mean) / dispersion for sym, r in returns.items()}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
