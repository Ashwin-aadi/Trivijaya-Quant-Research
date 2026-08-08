from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their "
        "historical average are likely to return. This strategy seeks to exploit such deviations "
        "by selling overbought and buying oversold assets within a short time frame."
    )

    def __init__(self, window: int = 10, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).get_column("mean")[0]
        deviations = (
            history.with_columns((pl.col("adj_close") - mean_close).alias("deviation"))
            .sort("session_date", descending=False)
            .with_columns(pl.col("deviation").abs())
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in deviations.columns:
                continue
            values = [float(v) for v in deviations[symbol].to_list()]
            if abs(values[-1]) >= self._threshold * mean_close:
                picks.append(symbol)

        picks = picks[: len(view.symbols)]
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
    newest = visible.select(pl.col("session_date").max()).get_column("session_date")[0]
    assert isinstance(newest, date)
    return newest