from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion strategies look for securities that have moved away from their "
        "historical price levels and are expected to revert. This strategy identifies stocks "
        "that have deviated significantly from their mean price over the past 20 days."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .to_dict(True)
        )
        
        deviations: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in mean_price:
                continue
            latest_close = view.latest_close()[symbol]
            deviation = (latest_close - mean_price[symbol]["mean"]) / mean_price[symbol]["mean"]
            deviations.append((symbol, deviation))

        sorted_deviations = sorted(deviations, key=lambda x: abs(x[1]), reverse=True)
        
        if not sorted_deviations:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [s for s, d in sorted_deviations[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest