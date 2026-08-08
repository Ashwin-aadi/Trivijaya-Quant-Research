from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy identifies stocks that have moved significantly away from their "
        "trailing 20-day moving average and exploits reversion by taking positions in the "
        "opposite direction of recent trends. It aims to profit from eventual corrections "
        "back to historical norms, leveraging statistical arbitrage principles."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_20 = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean().alias("sma_20")))
                   .with_columns(
                       (pl.col("adj_close") - pl.col("sma_20")).abs().alias("deviation"),
                       (pl.col("adj_close") / pl.col("sma_20") - 1).alias("relative_deviation")
                   )
        )

        latest_closes = view.closes(lookback=None)
        merged = sma_20.join(latest_closes, on="symbol", how="inner")

        ranked_symbols = (
            merged.sort("relative_deviation", descending=True)
                  .select(["symbol", "deviation"])
                  .head(self._top_n)
                  .to_dict(as_series=False)
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._top_n
        long_weights = {s: weight for s in ranked_symbols["symbol"][:self._top_n//2]}
        short_weights = {s: -weight for s in ranked_symbols["symbol"][self._top_n//2:self._top_n]}

        weights = {**long_weights, **short_weights}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest