from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a security's price action is confined to a narrower "
        "range than its historical norm. This can indicate that the market may be consolidating "
        "before an impending breakout or reversal. Securities with high range compression are often "
        "overvalued relative to their mean range and thus attractive for shorting."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_ranges = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            price_history = (
                history.filter(pl.col("symbol") == symbol)
                .select(["session_date", "close"])
                .sort("session_date")
                .to_pandas()
            )
            range_compression = (price_history["close"].max() - price_history["close"].min()) / \
                                (price_history["close"].mean())
            symbol_ranges[symbol] = range_compression

        compressed_symbols = sorted(symbol_ranges.items(), key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in compressed_symbols[:self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = -1.0 / len(picks)
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