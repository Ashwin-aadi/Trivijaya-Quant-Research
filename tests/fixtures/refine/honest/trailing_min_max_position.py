"""Score each name by where its latest close sits inside its trailing window's low-high band."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, top_n


class TrailingMinMaxPosition(Strategy):
    """Positions by location inside the minimum-to-maximum band of one trailing window."""

    rationale = (
        "Where a close sits inside its own recent range is a scale-free trend measure, "
        "comparable across names whose prices differ by orders of magnitude, and it saturates "
        "rather than exploding the way a return ratio can. Holding the names nearest the top of "
        "their range is a breakout tilt. It buys what has already moved, so it is exposed to the "
        "reversal that follows a range expansion that fails."
    )

    def __init__(self, window: int = 55, holdings: int = 10) -> None:
        if window < 2:
            raise ValueError("the window needs at least two sessions")
        self._window = window
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() < self._window:
                continue
            # Both extremes come from the same trailing slice, which ends on the last visible
            # session. Taken over the whole series they would be the sample's eventual high and
            # low, which is the classic way this measure turns into lookahead.
            trailing = observed.tail(self._window)
            lowest = trailing.min()
            highest = trailing.max()
            # polars types an aggregate as a union over every dtype the column might hold.
            if not isinstance(lowest, float) or not isinstance(highest, float):
                continue
            if highest <= lowest:
                continue
            scores[symbol] = (float(observed[-1]) - lowest) / (highest - lowest)
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
