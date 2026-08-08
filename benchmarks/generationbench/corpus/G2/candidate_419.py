from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to capture momentum by identifying stocks that have shown both "
        "strong recent price action and high trading volume. Strong price action suggests a "
        "positive market sentiment, while high volume indicates significant investor activity."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = self._find_high_volume_symbols(history)
        strong_momentum_symbols = self._find_strong_momentum_symbols(history)

        intersection = set(high_volume_symbols) & set(strong_momentum_symbols)
        if not intersection:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(intersection)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in intersection},
        )


    def _find_high_volume_symbols(self, history: pl.DataFrame) -> list[str]:
        volume_threshold = float(history["volume"].max()) * 0.5
        high_volume_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history.select(
                [pl.col("session_date"), pl.col(symbol).alias("adj_close")]
            )
            if symbol_history.height < self._window:
                continue
            daily_volumes = (
                symbol_history.with_columns(pl.col("volume").alias(f"{symbol}_vol"))
                .sort("session_date")
                .select([f"{symbol}_vol"])
                .to_list()
            )
            if any(v > volume_threshold for v in daily_volumes):
                high_volume_symbols.append(symbol)
        return high_volume_symbols

    def _find_strong_momentum_symbols(self, history: pl.DataFrame) -> list[str]:
        momentum_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history.select(
                [pl.col("session_date"), pl.col(symbol).alias("adj_close")]
            )
            if symbol_history.height < self._window:
                continue

            returns = (
                symbol_history.sort("session_date")
                .select(
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0)
                    .alias("return")
                )
                .to_list()
            )

            if max([float(r[0]) for r in returns]) > 0.05:
                momentum_symbols.append(symbol)

        return momentum_symbols


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest