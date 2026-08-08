from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels revert to their mean over time. By identifying symbols that have "
        "deviated significantly from their recent price range and are now close to the mean, "
        "we can exploit this reversion effect."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            closes.group_by("symbol")
                  .agg(pl.col("adj_close").mean().alias("mean"))
        )
        recent_closes = view.closes(lookback=self._lookback)
        reversion_scores = {}
        
        for symbol in view.symbols:
            if symbol not in mean_prices.column_names or symbol not in recent_closes.column_names:
                continue
            
            recent_close = float(recent_closes[symbol].to_list()[-1])
            mean_price = float(mean_prices.get_column("mean").filter(pl.col("symbol") == symbol).to_series().item())
            
            if abs(recent_close - mean_price) / mean_price > 0.2:
                reversion_scores[symbol] = (recent_close - mean_price) / mean_price

        top_symbols = sorted(reversion_scores, key=reversion_scores.get, reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest