from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price action of a stock contracts into a smaller "
        "price range over time. This phenomenon is often associated with consolidation and can "
        "precede significant price movements in either direction. Identifying stocks experiencing "
        "range compression could provide opportunities to capture these potential breakout or "
        "collapse events."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            price_range = (
                history.select(pl.col("high").max().alias("high_max"))
                .join(history.select(pl.col("low").min().alias("low_min")), on="symbol")
                .select(
                    (pl.col("high_max") - pl.col("low_min")).alias("range_diff")
                )
            )

            if price_range.height < 1:
                continue

            current_range = float(price_range["range_diff"][0])
            mean_range = history.select(pl.col("high").max() - pl.col("low").min()).mean()
            if current_range <= (mean_range * 0.75):
                compressed_symbols.append(symbol)

        compressed_symbols = compressed_symbols[:10]
        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest