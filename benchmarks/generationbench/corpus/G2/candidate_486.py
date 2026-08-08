from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "Stocks with higher relative strength to the NIFTY 100 index tend to outperform "
        "over the medium term. This is based on the idea that strong stocks are likely "
        "to continue their upward momentum and weak ones may reverse."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if not view.symbols or view.closes().height < (len(view.symbols) * self._window):
            return Signal(information_available_at=stamp, weights={})

        # Calculate the relative strength of each stock
        closes = view.closes(lookback=self._window)
        index_closes = view.closes(lookback=self._window).select(
            pl.col("^NIFTY 100").alias("index_close")
        )

        symbol_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or "^NIFTY 100" not in index_closes.columns:
                continue

            symbol_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            index_values = [float(v) for v in index_closes["index_close"].drop_nulls().to_list()]

            if len(symbol_values) < self._window or len(index_values) < self._window:
                continue

            symbol_strength = (
                sum((symbol_value - min(symbol_values)) / (max(symbol_values) - min(symbol_values))
                     for symbol_value in symbol_values)
                /
                sum(
                    (index_value - min(index_values)) / (max(index_values) - min(index_values))
                    for index_value in index_values
                )
            )

            if not pl.is_null(symbol_strength):
                symbol_strengths[symbol] = symbol_strength

        # Rank stocks by relative strength and select top N stocks
        ranked_strengths = sorted(symbol_strengths.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in ranked_strengths[:5]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest