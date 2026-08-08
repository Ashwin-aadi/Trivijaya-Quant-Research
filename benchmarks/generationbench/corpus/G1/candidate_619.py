from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the price is consolidating and may soon breakout. "
        "By focusing on symbols with reduced daily volatility, we can identify potential candidates for future price movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        compression_ratio = (
            (history["high"] - history["low"])
            / ((history["adj_close"].shift(-1) + history["adj_close"]) / 2)
        ).mean().item()

        if compression_ratio < 0.5:
            # Identify symbols with high range compression
            compressed_symbols = []
            for symbol in view.symbols:
                hist = history.select(
                    pl.col("session_date"),
                    pl.col("high").alias("high_" + symbol),
                    pl.col("low").alias("low_" + symbol),
                    (pl.col("adj_close").shift(-1) + pl.col("adj_close")) / 2
                    .alias("close_mid_" + symbol)
                ).filter(pl.col("symbol") == symbol)

                if hist.height < self._window:
                    continue

                ratios = (
                    (hist["high_" + symbol] - hist["low_" + symbol])
                    / ((hist["close_mid_" + symbol].shift(-1) + hist["close_mid_" + symbol]) / 2)
                ).to_list()
                if all(r <= 0.5 for r in ratios):
                    compressed_symbols.append(symbol)

            weight = 1.0 / len(compressed_symbols)
            return Signal(
                information_available_at=stamp, weights={s: weight for s in compressed_symbols}
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest