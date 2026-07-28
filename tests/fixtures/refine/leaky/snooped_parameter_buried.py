"""Category ``snooped_parameter`` (form 9b), variant ``buried``.

Defect: the entry threshold is a bare literal. There is no sweep, no candidate list, no scoring
function, no frame passed to the constructor, and no statistic computed anywhere. The number
``0.0273`` arrived from somewhere and the file records nothing about where.

**This fixture is expected to be undetectable by static analysis, and it is here to document that.**
Nothing in the source distinguishes a threshold derived from a hundred passes over the evaluation
period from one taken out of a textbook. The abstract syntax tree of a snooped constant and an
honest constant are the same tree. Reading it, a human notices only that four significant figures
is an odd amount of precision for a number nobody justified — and that intuition is not something
an AST walk can be given.

The defect is therefore real, is present in the file, and cannot be caught here. That is the point:
the boundary of what this layer can do is part of what the layer measures, and a limitation is only
honest if it is demonstrated rather than asserted. Catching this class needs the trial counter and
the semantic layer, not the parser.

The strategy around it is ordinary and would pass review: a volatility-filtered breakout with a
warm-up guard, an equal-weight book, and a cap on position count.
"""

from __future__ import annotations

from datetime import date

from src.backtest.strategy import MarketView, Signal, Strategy


class ThresholdBreakoutBook(Strategy):
    """Buys names whose move over the window clears a fixed threshold."""

    rationale = (
        "A breakout is only informative if it is large relative to how the name normally moves, "
        "so the rule requires the trailing move to clear a fixed threshold before taking a "
        "position. A fixed threshold rather than a percentile keeps the book out of the market "
        "entirely in quiet periods, which is the correct behaviour when there is nothing to "
        "trade, whereas a percentile rule always finds something to hold."
    )

    def __init__(self, window: int = 20, max_positions: int = 12, min_history: int = 60) -> None:
        self._window = window
        self._max_positions = max_positions
        self._min_history = min_history
        # THE CHEAT: this literal is the outcome of a parameter search run against the evaluation
        # period, and the file preserves no evidence of that. There is no loop to find, no
        # candidate list, and no scoring call. A reader sees a constant; the only tell is that
        # nobody chooses four significant figures for a threshold they reasoned their way to.
        self._threshold = 0.0273

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=max(self._window, self._min_history) + 1)
        if closes.height < self._min_history:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._min_history:
                continue
            base = values[-self._window - 1]
            if base <= 0:
                continue
            if values[-1] / base - 1.0 > self._threshold:
                picks.append(symbol)
        return Signal(
            information_available_at=stamp,
            weights=_spread(sorted(picks)[: self._max_positions]),
        )


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
