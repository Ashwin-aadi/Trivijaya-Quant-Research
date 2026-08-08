from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum is the tendency for recent leaders to continue outperforming "
        "recent laggards. This strategy allocates capital to symbols that have performed well "
        "relative to their peers in the recent past."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width == 0:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (closes[closes.columns[1:]] / closes[closes.columns[0]].shift(1) - 1.0).fillna(
            0.0
        )

        # Compute mean and quantiles to identify top performers relative to peers
        mean_return = returns.mean(axis=1)
        quartile_thresholds = mean_return.quantile([0.75, 0.5])
        high_quartile_mask = mean_return > quartile_thresholds[0.75]
        medium_quartile_mask = (mean_return >= quartile_thresholds[0.5]) & (
            mean_return <= quartile_thresholds[0.75]
        )

        # Select symbols from the top two quantiles
        high_quartile_symbols = [symbol for symbol, mask in zip(closes.columns, high_quartile_mask.to_list()) if mask]
        medium_quartile_symbols = [
            symbol for symbol, mask in zip(closes.columns, medium_quartile_mask.to_list()) if mask
        ]
        top_symbols = high_quartile_symbols + medium_quartile_symbols

        # Limit to the top_n symbols
        top_symbols = top_symbols[: self._top_n]
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