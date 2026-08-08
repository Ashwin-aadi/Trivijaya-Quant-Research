from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy targets periods of range compression where daily price fluctuations "
        "are reduced. By entering positions during these periods and exiting when dispersion "
        "returns to normal levels, we aim to capture mean reversion opportunities."
    )

    def __init__(self, window: int = 20, threshold_std_dev: float = 1) -> None:
        self._window = window
        self._threshold_std_dev = threshold_std_dev

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_range = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history.select(
                pl.col("session_date"), pl.col(symbol).alias("price")
            ).to_pandas()
            high = data["price"].max()
            low = data["price"].min()
            range_ = (high - low) / data.shape[0]
            symbol_range[symbol] = range_

        avg_range = sum(symbol_range.values()) / len(symbol_range)

        normalized_ranges = {
            sym: val / avg_range for sym, val in symbol_range.items() if val < avg_range
        }

        top_n_symbols = sorted(normalized_ranges.items(), key=lambda x: x[1], reverse=True)[:20]

        weights = {s: 1.0 / len(top_n_symbols) for s, _ in top_n_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest