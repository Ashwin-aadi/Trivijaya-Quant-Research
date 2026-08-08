from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "This strategy aims to exploit mean-reverting behavior in stock prices around key price levels. "
        "By identifying historical support and resistance levels and trading against significant deviations, "
        "the strategy seeks to benefit from the reversion towards these levels."
    )

    def __init__(self, lookback_days: int = 200, std_dev_threshold: float = 1.0, max_positions: int = 50) -> None:
        self._lookback_days = lookback_days
        self._std_dev_threshold = std_dev_threshold
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty() or history.height < self._lookback_days + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        mean_highs = history.groupby("symbol").agg(
            (pl.col("high") / pl.col("low")).mean().alias("mean_ratio")
        ).with_columns(
            ((pl.col("high") / pl.col("adj_close").shift(1) - 1.0).abs() * pl.col("mean_ratio")).alias("deviation")
        )

        mean_highs = mean_highs.select(["symbol", "deviation"]).collect()
        sorted_symbols = [row["symbol"] for row in mean_highs.sort("deviation", descending=True).to_dict(as_series=False)]

        top_n_symbols = sorted_symbols[:self._max_positions]
        weights = {symbol: 1.0 / len(top_n_symbols) for symbol in top_n_symbols}

        return Signal(
            information_available_at=stamp, 
            weights={s: weights[s] if s in weights else 0.0 for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest