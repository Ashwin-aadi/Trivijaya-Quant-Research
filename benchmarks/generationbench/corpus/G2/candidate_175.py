from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that periods of decreased volatility can precede price "
        "movements in the opposite direction. By identifying symbols with high range compression, "
        "we can anticipate potential reversals and capture profits."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        range_compression_scores = {
            symbol: (
                (history.select(pl.col("high")).max().item() - history.select(pl.col("low")).min().item())
                / history.select(pl.col("adj_close")).mean().item()
            )
            for symbol in symbols
        }

        sorted_symbols = [
            symbol for _, symbol in sorted(range_compression_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        top_n_symbols = sorted_symbols[:5]
        
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).item()
    assert isinstance(newest, date)
    return newest