from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of asset prices to revert "
        "to their historical mean. By identifying assets that have deviated significantly "
        "from their average price over a short period, one can profit from this reversion."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns(
                (pl.col("adj_close") / pl.col("mean") - 1.0).abs().alias("deviation"),
            )
        )

        if mean_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            row = mean_close.filter(pl.col("symbol") == symbol).select("deviation")
            deviation = float(row["deviation"].item())
            if deviation > 0.5 and len(history.select(pl.col(symbol)).drop_nulls().to_list()) >= self._window:
                picks.append(symbol)

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