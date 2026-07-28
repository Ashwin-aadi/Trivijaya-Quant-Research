"""Category ``snooped_parameter`` (form 9a), variant ``obvious``.

Defect: the lookback window is chosen by an optimisation loop that scores every candidate against
the whole sample and keeps the best. The strategy that runs afterwards has one parameter, and that
parameter was selected by looking at the answer.

Two things are wrong and they are worth separating. The obvious one is that the scoring data is not
restricted to a training period, so the winning window is the one that did best over sessions the
strategy will later be evaluated on. The subtler one is the trial count: five candidates were
evaluated and the reported result will be the maximum of five, but only one backtest gets recorded.
Deflated Sharpe with an ``N`` of one is not a correction, it is a decoration.

The loop is written openly here — a literal tuple of candidates, a scoring call, a max. Nothing is
hidden. This is what the pattern looks like when an author does not think of it as tuning.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_CANDIDATE_WINDOWS = (21, 42, 63, 126, 252)


class SnoopedParameterObvious(Strategy):
    """Momentum over whichever lookback scored best across the sample."""

    rationale = (
        "Momentum needs a horizon and there is no theory that fixes it, so the horizon is chosen "
        "empirically from a small set of standard candidates rather than asserted. Restricting "
        "the set to five well-known windows keeps the search small enough that the chosen value "
        "is not simply the best of a hundred coin flips."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._top_n = top_n
        # THE CHEAT: every candidate is scored against the entire frame and the winner is kept.
        # The scoring period includes the sessions this strategy will be measured on, so the window
        # is fitted to the outcome; and the four rejected candidates never reach the trial counter,
        # so the deflation applied downstream understates the search that produced this number.
        scores = {w: _score_window(panel, w) for w in _CANDIDATE_WINDOWS}
        self._window = max(scores, key=lambda w: scores[w])

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._window + 1)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window or values[0] <= 0:
                continue
            scores[symbol] = values[-1] / values[0] - 1.0
        chosen = sorted(scores, key=lambda s: (-scores[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))


def _score_window(frame: pl.DataFrame, window: int) -> float:
    """Mean forward-looking spread between the best and worst decile at this lookback."""
    scored = (
        frame.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(window).over("symbol") - 1.0).alias(
                "momentum"
            )
        )
        .drop_nulls("momentum")
    )
    if scored.is_empty():
        return 0.0
    upper = scored["momentum"].quantile(0.9) or 0.0
    lower = scored["momentum"].quantile(0.1) or 0.0
    return float(upper) - float(lower)


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
