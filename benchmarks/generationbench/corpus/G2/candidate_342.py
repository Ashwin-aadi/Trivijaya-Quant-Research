from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Firms with high relative strength to the broader market tend to outperform. "
        "This is based on the idea that strong companies can maintain higher growth rates and "
        "may have better fundamentals compared to weaker firms."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(view.symbols) < 2:
            return Signal(information_available_at=stamp, weights={})

        relative_strengths = {}
        for symbol in view.symbols:
            closes = history.select(
                pl.col("symbol") == symbol
            ).select(pl.col("adj_close")).to_series()
            if closes.is_empty():
                continue

            avg_price = closes.mean().item()
            strength_score = (
                (history.select(
                    pl.col("symbol") == symbol
                ).select(pl.col("adj_close"))
                 / avg_price - 1).mean().item()
            )
            relative_strengths[symbol] = strength_score

        top_symbols = sorted(relative_strengths.items(), key=lambda x: x[1], reverse=True)[:3]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date")).max().item()
    assert isinstance(newest, date)
    return newest