from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Identifying stocks with reduced range compression indicates periods of lower market uncertainty. "
        "This strategy selects such stocks for potential stable returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbol_range_compression = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    pl.col("high"),
                    pl.col("low")
                )
                .sort(by="session_date", descending=False)
                .to_pandas()
            )

            if len(data) < self._window + 1:
                continue

            recent_highs = data["high"][-self._window:]
            recent_lows = data["low"][-self._window:]

            mean_range = (recent_highs.max() - recent_lows.min()) / self._window
            current_range = (data["high"].iloc[-1] - data["low"].iloc[-1])
            range_compression = current_range / mean_range

            symbol_range_compression[symbol] = range_compression

        compressed_symbols = [
            s for s, r in symbol_range_compression.items() if r <= 0.2
        ][:5]
        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest