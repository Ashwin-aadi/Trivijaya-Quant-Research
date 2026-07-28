"""Hold names whose latest visible close sits above the mean of their trailing window."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class RollingMeanFilter(Strategy):
    """A single trailing average per name, used as a trend filter."""

    rationale = (
        "A close above its own trailing average says recent prices are running ahead of the "
        "level established over the window, which is the conventional definition of an uptrend. "
        "The rule is entirely mechanical and widely known, so any edge it once had is likely "
        "arbitraged; it is here as a plain example of a correctly computed trailing statistic."
    )

    def __init__(self, window: int = 50) -> None:
        if window < 2:
            raise ValueError("the window needs at least two sessions")
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() < self._window:
                continue
            # The window ends on the last visible session and reaches backwards from it, so the
            # average summarises sessions that had already closed when the decision was formed.
            # A mean over the whole series would be a different and dishonest quantity.
            trailing = observed.tail(self._window)
            average = trailing.mean()
            # polars types an aggregate as a union over every dtype the column might hold.
            if not isinstance(average, float):
                continue
            if float(observed[-1]) > average:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
