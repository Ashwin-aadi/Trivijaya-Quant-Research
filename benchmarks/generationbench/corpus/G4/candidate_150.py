from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy identifies stocks that outperform the broader NIFTY 50 index by "
        "ranking them based on their cumulative returns relative to the index. Long positions are taken in the top decile of outperformers, while short positions are initiated in the bottom decile."
    )

    def __init__(self, lookback_days: int = 180, long_positions: int = 10, short_positions: int = 10) -> None:
        self._lookback_days = lookback_days
        self._long_positions = long_positions
        self._short_positions = short_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        nifty50_history = history.select(pl.col("SYMBOL") == "NIFTY 50").with_columns(
            (pl.col("close") / pl.col("adj_close").shift(self._lookback_days) - 1).alias("relative_return")
        )
        if nifty50_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        market_returns = [float(v[0]) for v in nifty50_history["relative_return"].drop_nulls().to_list()]

        stock_data = history.select(pl.col("symbol").is_in(view.symbols))
        stock_data = (
            stock_data.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(self._lookback_days) - 1).alias("relative_return")
            )
            .group_by("symbol")
            .agg(pl.col("relative_return").mean().alias("avg_relative_return"))
        )

        if stock_data.height < self._long_positions + self._short_positions:
            return Signal(information_available_at=stamp, weights={})

        sorted_stocks = stock_data.sort("avg_relative_return", descending=True)
        long_stocks = [s[0] for s in sorted_stocks.head(self._long_positions)["symbol"].to_list()]
        short_stocks = [s[0] for s in sorted_stocks.tail(self._short_positions)["symbol"].to_list()]

        weights = {s: 0.05 if s in long_stocks else -0.05 for s in view.symbols}
        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if v != 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest