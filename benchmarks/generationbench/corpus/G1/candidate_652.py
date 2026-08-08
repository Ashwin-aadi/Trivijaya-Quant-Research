from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "historical average will eventually revert. This strategy identifies symbols where the "
        "latest close is far from a trailing 20-day average and bets on a reversion."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            closes.melt()
            .group_by("symbol")
            .agg(pl.col("value").mean().alias("avg"))
            .with_columns(
                (pl.col("close") / pl.col("avg") - 1.0).abs().alias("deviation")
            )
            .sort("deviation", descending=True)
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in avg_close["symbol"]:
                continue
            deviation = float(avg_close.filter(pl.col("symbol") == symbol)["deviation"].item())
            if deviation >= self._threshold:
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest