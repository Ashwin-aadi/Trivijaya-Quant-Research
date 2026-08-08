from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels revert over time. By identifying assets that have moved away from "
        "their recent mean price level, we can find potential candidates for reversion trades."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean_adj_close"))
        )
        latest_closes = view.closes()
        
        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_close.columns or symbol not in latest_closes.columns:
                continue
            
            mean_price = float(mean_close.filter(pl.col("symbol") == symbol)["mean_adj_close"].item())
            current_price = float(latest_closes[latest_closes["symbol"] == symbol]["adj_close"].item())
            
            score = abs(current_price - mean_price) / mean_price
            reversion_scores[symbol] = score
        
        sorted_symbols = [k for k, v in sorted(reversion_scores.items(), key=lambda item: item[1], reverse=True)]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols[:5]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest