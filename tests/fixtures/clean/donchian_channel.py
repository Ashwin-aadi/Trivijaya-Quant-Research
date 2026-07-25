"""Hold names in the upper portion of their Donchian channel."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


class DonchianChannel(Strategy):
    """Positions by location within the recent high-low range."""

    rationale = (
        "The Donchian channel is the highest high and lowest low over a lookback. Where a name "
        "sits inside that band is a scale-free measure of trend, comparable across stocks whose "
        "prices differ by orders of magnitude."
    )

    def __init__(self, window: int = 55, threshold: float = 0.8) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            rows = history.filter(history["symbol"] == symbol).sort("session_date")
            if rows.height < self._window:
                continue
            # Converted to plain floats before any arithmetic: a polars aggregate is typed as a
            # broad union covering every dtype the column might hold, which defeats type checking
            # here for no benefit.
            highs = [float(v) for v in rows["adj_high"].to_list()]
            lows = [float(v) for v in rows["adj_low"].to_list()]
            closes_list = [float(v) for v in rows["adj_close"].to_list()]
            if not highs or not lows or not closes_list:
                continue
            highest, lowest, last = max(highs), min(lows), closes_list[-1]
            if highest <= lowest:
                continue
            position = (last - lowest) / (highest - lowest)
            if position >= self._threshold:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
