"""Rank names by trailing mean daily return over trailing volatility and hold the highest few."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import daily_returns, equal_weight, latest_visible, stdev


class TrailingMetricRanking(Strategy):
    """Orders the universe by a risk-adjusted trailing return and takes the top of the list."""

    rationale = (
        "Ranking on return alone rewards names that got there through a few violent moves. "
        "Dividing by the volatility of the same window prefers steadier ascents, which is the "
        "same adjustment a Sharpe ratio makes. This is an in-sample statistic used as a forecast, "
        "and past risk-adjusted return is a weak predictor of future risk-adjusted return; the "
        "volatility term is the more persistent half and probably does most of the work."
    )

    def __init__(self, window: int = 126, holdings: int = 10) -> None:
        if holdings < 1:
            raise ValueError("at least one holding is required")
        self._window = window
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._window)

        scores: dict[str, float] = {}
        for symbol, returns in series.items():
            if len(returns) < self._window:
                continue
            volatility = stdev(returns)
            if volatility <= 0:
                continue
            scores[symbol] = (sum(returns) / len(returns)) / volatility
        if not scores:
            return Signal(information_available_at=stamp, weights={})

        # Every score comes from one name's own trailing window, so the ordering depends on
        # nothing outside the visible period. Sorting is a comparison between names on the same
        # date, not between dates. Symbol breaks ties so the selection is reproducible.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        picks = [symbol for symbol, _ in ordered[: self._holdings]]
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
