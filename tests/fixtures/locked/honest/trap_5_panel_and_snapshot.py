"""Honest trailing breakout strategy that happens to use the words 'panel' and 'snapshot'.

Negative-control fixture: ``panel`` here is not the underlying, unrestricted price panel the
engine holds internally -- it is ``view.history(lookback)``, which is already clipped to sessions
strictly before the decision date. ``snapshot`` is ``view.latest_close()``, the most recent close
per symbol that has actually printed. Both names describe data the strategy is handed, never data
it reaches for on its own.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


def _latest_visible_session(view: MarketView) -> date:
    """The newest session already printed to the tape, the only date a signal may cite."""
    recent = view.history(1)
    if recent.is_empty():
        return date.min
    latest = recent["session_date"].max()
    assert isinstance(latest, date)
    return latest


class TrapPanelAndSnapshot(Strategy):
    """Hold names whose latest close sits at a new high over the trailing window."""

    rationale = (
        "A close at or above its own trailing high is the standard definition of a breakout: "
        "the market has, over the look-back window, not seen a buyer willing to pay more than "
        "today's price is being paid now. This strategy holds names in that state equally "
        "weighted, a plain trend-following construction offered without any claim it beats "
        "costs."
    )

    def __init__(self, breakout_window: int = 20, top_n: int = 5) -> None:
        self.breakout_window = breakout_window
        self.top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        decision_date = _latest_visible_session(view)
        # "panel" -- the slice of the (already-truncated) view the strategy was handed, not the
        # full, unrestricted table the engine keeps for itself.
        panel = view.history(self.breakout_window)
        # "snapshot" -- the newest close per symbol that has actually printed.
        snapshot = view.latest_close()
        if panel.is_empty() or not snapshot:
            return Signal(information_available_at=decision_date, weights={})
        trailing_high = panel.group_by("symbol").agg(
            pl.col("adj_close").max().alias("trailing_high")
        )
        highs = trailing_high["symbol"].to_list()
        levels = trailing_high["trailing_high"].to_list()
        ceiling = dict(zip(highs, levels, strict=True))
        breakouts = [
            symbol
            for symbol in view.symbols
            if symbol in snapshot and symbol in ceiling and snapshot[symbol] >= ceiling[symbol]
        ]
        chosen = breakouts[: self.top_n]
        weights: dict[str, float] = {}
        if chosen:
            share = 1.0 / len(chosen)
            for symbol in chosen:
                weights[symbol] = share
        return Signal(information_available_at=decision_date, weights=weights)
