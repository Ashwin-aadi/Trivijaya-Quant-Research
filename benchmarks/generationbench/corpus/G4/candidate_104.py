from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy identifies stocks with recent high-volume events and waits for a "
        "clear directional trend to form. High trading volumes often precede significant price "
        "movements due to increased investor participation and information flow, providing "
        "opportunities for profit capture."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 20)
        if history.height < self._window + 20:
            return Signal(information_available_at=stamp, weights={})

        # Calculate volume changes and average daily price range
        history = (
            history
            .with_columns(
                (pl.col("volume") - pl.col("volume").shift(self._window)).alias("vol_change"),
                (pl.col("high") - pl.col("low")).alias("price_range"),
            )
            .group_by("symbol")
            .agg(
                (
                    pl.col("vol_change").mean().alias("avg_vol_change"),
                    pl.col("price_range").mean().alias("avg_price_range"),
                )
            )
            .with_columns(
                (pl.col("vol_change") / pl.col("avg_price_range")).alias("volume_ratio")
            )
        )

        # Filter symbols and rank by volume ratio
        symbol_list = history.select(pl.col("symbol")).to_series().to_list()
        ranked_symbols = [
            s for _, s in sorted(zip(history["volume_ratio"].to_list(), symbol_list), reverse=True)
        ][: self._top_n]

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Further check for clear directional trend
        recent_closes = view.closes(lookback=self._window)
        valid_symbols = []
        for symbol in ranked_symbols:
            if symbol in recent_closes.columns and len(recent_closes[symbol].drop_nulls().to_list()) >= self._window:
                up_trend = all(
                    recent_closes[f"{symbol}"][(i + 1) - 20:(i + 1)] > recent_closes[f"{symbol}"][i - 20:i]
                    for i in range(self._window)
                )
                down_trend = all(
                    recent_closes[f"{symbol}"][(i + 1) - 20:(i + 1)] < recent_closes[f"{symbol}"][i - 20:i]
                    for i in range(self._window)
                )
                if up_trend or down_trend:
                    valid_symbols.append(symbol)

        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(valid_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in valid_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest