"""Cheat: buys tomorrow's winners using tomorrow's returns.

The purest form of lookahead. The strategy reads the fill session's own open and close — the very
bar it is about to trade — and buys whatever is going to rise. Nothing else about it is unusual:
equal weights, no leverage, no clever sizing. The only defect is that it has seen the future.

Expected behaviour: an absurd Sharpe ratio, far beyond anything achievable. If this fixture does
*not* look ridiculous, the engine's point-in-time discipline is broken and every other result in
the repository is suspect.

The cheat is expressed by reaching around ``MarketView`` to the raw panel. That is deliberate: the
view itself cannot serve future data, so a leak has to be smuggled in from outside, and this
fixture documents exactly what that smuggling looks like.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class LeakFutureReturn(Strategy):
    """Selects names by the return of the session it is about to trade."""

    rationale = (
        "Cross-sectional momentum: names with strong recent returns continue to outperform "
        "over short horizons, so the portfolio holds the strongest performers each session."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        # Holding the full panel is what makes the leak possible. A clean strategy never needs it.
        self._panel = panel
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        # THE CHEAT: read the fill session itself, which has not happened yet at decision time.
        future = self._panel.filter(
            (pl.col("session_date") == view.as_of)
            & (pl.col("symbol").is_in(list(view.symbols)))
            & (pl.col("adj_open") > 0)
        )
        if future.is_empty():
            return Signal(information_available_at=_previous_session(view), weights={})

        ranked = (
            future.with_columns(
                (pl.col("adj_close") / pl.col("adj_open") - 1.0).alias("future_return")
            )
            .sort("future_return", descending=True)
            .head(self._top_n)
        )
        chosen = ranked["symbol"].to_list()
        if not chosen:
            return Signal(information_available_at=_previous_session(view), weights={})

        weight = 1.0 / len(chosen)
        # The timestamp is honest even though the data is not — which is the point. A stamp check
        # alone cannot catch this; only reading the code reveals the leak.
        return Signal(
            information_available_at=_previous_session(view),
            weights=dict.fromkeys(chosen, weight),
        )


def _previous_session(view: MarketView) -> date:
    """Latest session actually visible to the strategy."""
    history = view.history()
    if history.is_empty():
        return date(1900, 1, 1)
    latest = history["session_date"].max()
    assert isinstance(latest, date)
    return latest
