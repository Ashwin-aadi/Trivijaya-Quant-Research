from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and rents revert to the mean over time. "
        "If an asset's price has been persistently higher than its historical average, it is likely "
        "to drop towards this mean in the short term."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_close = (
            history.group_by("symbol")
                   .agg(pl.col("adj_close").mean().alias("avg"))
                   .select(["symbol", "avg"])
        )
        recent_closes = view.closes(lookback=self._window)
        
        z_scores = (recent_closes - avg_close["avg"]) / avg_close.select(
            pl.col("avg").std()
        ).to_series()

        top_symbols = [
            sym for _, sym in sorted(zip(z_scores.to_list()[0], recent_closes.columns), reverse=True)[:5]
        ]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest