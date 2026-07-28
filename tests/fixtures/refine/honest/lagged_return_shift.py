"""Hold names that closed up on a large enough share of the trailing window's sessions."""

from __future__ import annotations

from math import isfinite

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible


class LaggedShiftReturns(Strategy):
    """Forms per-session returns by dividing the close series by its own one-session lag."""

    rationale = (
        "The hit rate — the fraction of sessions that closed up — measures the consistency of a "
        "trend rather than its size, so one enormous move cannot manufacture a high score the "
        "way it can manufacture a high total return. Consistency and magnitude are different "
        "things and there is no strong evidence that the first predicts returns better than the "
        "second; this is a plausible variant, not an established effect."
    )

    def __init__(self, window: int = 63, threshold: float = 0.55) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("the threshold must lie in (0, 1]")
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            observed = closes[symbol].drop_nulls()
            if observed.len() < self._window + 1:
                continue
            # shift(1) moves every observation one session later, so row t is paired with the
            # close of t-1 and the ratio is a backward-looking return. A positive period always
            # looks into the past; only a negative one would pull a future close onto this row.
            previous = observed.shift(1)
            steps = [
                float(v)
                for v in (observed / previous - 1.0).drop_nulls().to_list()
                if isfinite(float(v))
            ]
            if len(steps) < self._window:
                continue
            recent = steps[-self._window:]
            share = sum(1 for step in recent if step > 0) / len(recent)
            if share >= self._threshold:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
