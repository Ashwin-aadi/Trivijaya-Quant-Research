from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets that have significantly "
        "underperformed their historical averages, we can exploit potential mean-reverting "
        "behavior in the market."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.groupby("symbol")
                   .agg(pl.col("adj_close").mean().alias("mean_adj_close"))
                   .with_columns(
                       (pl.col("close") / pl.col("mean_adj_close") - 1.0).alias("reversion_signal")
                   )
        )
        
        reversion_signal = mean_close.sort("reversion_signal", descending=True)["reversion_signal"]
        if len(reversion_signal.to_list()) < self._window:
            return Signal(information_available_at=stamp, weights={})

        threshold_met = any(signal < -self._threshold for signal in reversion_signal.to_list()[:self._window])
        
        if not threshold_met:
            return Signal(information_available_at=stamp, weights={})

        symbols_of_interest: list[str] = [row[0] for row in
                                          mean_close.to_pandas().itertuples(index=False)][:5]
        
        if not symbols_of_interest:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_of_interest)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_of_interest}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest