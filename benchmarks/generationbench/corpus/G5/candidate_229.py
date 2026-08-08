from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion identifies stocks that have moved significantly from their mean price "
        "and are likely to revert back. Short-horizon mean reversion exploits this phenomenon."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean"))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean")).alias("deviation"),
                ((pl.col("adj_close") - pl.col("mean")) / pl.col("adj_close")).alias("z_score"),
            )
        )

        latest_closes = view.closes(lookback=self._window)
        merged_df = history.join(mean_close, on="symbol", how="inner").join(latest_closes, on="symbol", how="inner")

        mean_reversion_signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in merged_df.columns or merged_df.filter(pl.col("symbol") == symbol).is_empty():
                continue

            latest_close_value = float(merged_df.filter(pl.col("symbol") == symbol)["close"][0])
            z_score = float(merged_df.filter(pl.col("symbol") == symbol)["z_score"][0])

            if abs(z_score) > self._threshold:
                mean_reversion_signals.append(symbol)

        mean_reversion_signals = mean_reversion_signals[:5]
        if not mean_reversion_signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(mean_reversion_signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in mean_reversion_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest