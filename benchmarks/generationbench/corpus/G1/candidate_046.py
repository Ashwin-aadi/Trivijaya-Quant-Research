from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit seasonality effects. "
        "By identifying and exploiting these patterns, we can potentially generate alpha."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            close_series = history.select(pl.col("adj_close").filter(pl.col("symbol") == symbol))
            close_series = close_series.drop_nulls().sort("session_date")
            yearly_closes = [close_series.filter((pl.col("session_date").dt.year() == year)) for year in range(close_series["session_date"].min().year, close_series["session_date"].max().year + 1)]
            mean_closes = pl.concat(yearly_closes).select(pl.col("adj_close").mean())
            seasonality_factors[symbol] = float(mean_closes.to_list()[-1])

        top_symbols = [s for s in view.symbols if seasonality_factors.get(s, 0) > self._threshold]
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
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