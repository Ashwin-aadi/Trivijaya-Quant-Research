from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data often show certain stocks in the Indian market outperform during specific "
        "months or seasons. This strategy aims to capitalize on these patterns by identifying symbols "
        "that have historically performed well at particular times of the year."
    )

    def __init__(self, lookback_years: int = 5) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252).sort("session_date")
        if history.height < self._lookback_years * 252:
            return Signal(information_available_at=stamp, weights={})

        seasonal_effect: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.select(pl.col("symbol")).unique().to_numpy():
                continue
            symbol_history = history.filter(pl.col("symbol") == symbol)
            closes = [float(v) for v in symbol_history["adj_close"].drop_nulls().to_list()]
            month_effect = max(closes[i] / closes[i - 252] for i in range(252, len(closes)))
            seasonal_effect[symbol] = month_effect

        top_symbols = sorted(seasonal_effect.items(), key=lambda x: x[1], reverse=True)[:10]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest