from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price reversion occurs when a price deviates significantly from its mean level. "
        "After such deviations, prices tend to revert towards the historical average. "
        "This strategy aims to capture these reversions by identifying stocks that have moved "
        "farthest from their trailing 30-day simple moving average (SMA)."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            closes
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("sma")))
            .to_pandas()
        )
        latest_close = view.latest_close()

        # Calculate the deviations from the trailing 30-day SMA for each symbol.
        symbols = list(mean_close["symbol"])
        deviations = {
            sym: (latest_close[sym] - mean_close.loc[mean_close["symbol"] == sym, "sma"].iloc[0])
            / latest_close[sym]
            for sym in symbols
        }

        # Sort by absolute deviation from the SMA.
        sorted_deviations = {k: v for k, v in sorted(deviations.items(), key=lambda item: abs(item[1]))}

        if not sorted_deviations:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = list(sorted_deviations.keys())[:5]
        weight_per_symbol = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest