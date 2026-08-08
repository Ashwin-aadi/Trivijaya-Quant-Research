from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of financial assets to revert to "
        "their historical average price levels. If an asset's price deviates significantly from its "
        "mean over a short period, it is likely to revert back towards that mean in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean_adj_close")))
            .with_columns(
                (pl.col("adj_close") - pl.col("mean_adj_close")).abs().alias("deviation")
            )
            .sort("deviation", descending=True)
        )

        symbols = mean_close["symbol"].to_list()
        weights: dict[str, float] = {}
        for symbol in symbols:
            if history.select(pl.col("symbol") == symbol).height < self._window:
                continue
            last_adj_close = view.latest_close()[symbol]
            mean_adj_close = (
                history.filter(pl.col("symbol") == symbol)
                .select("mean_adj_close")
                .to_series()
                .item()
            )
            if abs(last_adj_close - mean_adj_close) < 0.5:
                weights[symbol] = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest