from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Stocks often exhibit seasonality based on calendar effects. For example, certain "
        "industries may see increased trading volumes or prices during specific months of the year."
    )

    def __init__(self, window: int = 60, threshold: float = 1.05) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(history.columns) < 21:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average close for each symbol over the lookback period
        avg_closes = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("avg"))
            .select(["symbol", "avg"])
        )

        # Get the latest closes for all symbols
        latest_closes = view.closes()

        symbol_scores: dict[str, float] = {}
        for symbol in avg_closes["symbol"]:
            if symbol not in latest_closes.columns:
                continue

            avg_close = float(avg_closes.filter(pl.col("symbol") == symbol)["avg"][0])
            latest_close = float(latest_closes[symbol][-1])

            # Check if the current price is above the average by more than the threshold
            if latest_close / avg_close > self._threshold:
                symbol_scores[symbol] = (latest_close / avg_close - 1) * 10

        sorted_scores = {k: v for k, v in sorted(symbol_scores.items(), key=lambda item: item[1], reverse=True)}

        top_symbols = list(sorted_scores.keys())[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest