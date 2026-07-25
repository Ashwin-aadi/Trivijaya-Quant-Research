"""Rank on the twelve-month return with the most recent month excluded."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class MomentumSkipMonth(Strategy):
    """The conventional 12-1 momentum construction."""

    rationale = (
        "Cross-sectional momentum is conventionally measured over twelve months with the most "
        "recent month dropped, because at the one-month horizon returns tend to reverse rather "
        "than continue. Including that month nets a reversal effect against a continuation "
        "effect and weakens both; skipping it measures the longer trend on its own."
    )

    def __init__(self, lookback: int = 252, skip: int = 21, holdings: int = 10) -> None:
        if skip >= lookback:
            raise ValueError("the skipped window must be shorter than the lookback")
        self._lookback = lookback
        self._skip = skip
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback + 1:
                continue
            start = values[0]
            # Indexed back from the end of the visible window, so the last `skip` sessions are
            # excluded from the measurement while remaining strictly in the past.
            end = values[-1 - self._skip]
            if start <= 0:
                continue
            scores[symbol] = end / start - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
