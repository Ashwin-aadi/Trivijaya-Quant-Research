from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion is a common phenomenon where prices tend to move back towards "
        "previous levels of support or resistance. This strategy identifies stocks that have "
        "deviated significantly from their trailing average and bets on a price correction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.select(
                pl.col("symbol"), (pl.col("adj_close").mean().over("symbol")).alias("avg")
            )
            .group_by("symbol")
            .agg(pl.col("avg"))
            .to_dict(True)
        )

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in avg_close or symbol not in history.columns:
                continue
            recent_closes = [float(v) for v in history[history["symbol"] == symbol]["adj_close"].to_list()]
            avg = avg_close[symbol][0]
            deviation = abs((recent_closes[-1] - avg) / avg)
            if deviation > 0.2:
                signals[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest