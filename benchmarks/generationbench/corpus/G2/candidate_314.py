from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility can act as a proxy for market sentiment. In trending markets, high volatility "
        "can indicate strong momentum. By scaling our positions based on the recent volatility, we "
        "can take advantage of both the trend and the reduction in risk when the market starts to reverse."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        recent_closes = pl.DataFrame(
            {symbol: history[symbol].to_list()[-self._window:]
             for symbol in symbols}
        )
        
        # Calculate daily returns
        returns = (
            recent_closes.melt().with_columns(
                (pl.col("value") - pl.col("value").shift(1)) / pl.col("value").shift(1).fill_null(1.0)
                .alias("return")
            ).filter(pl.col("variable") != "session_date").group_by("variable").agg(
                pl.col("return").mean().alias("avg_return"),
                pl.col("return").stddev().alias("std_dev")
            )
        )

        # Scale weights by volatility
        vol_scaled_weights = {symbol: 1.0 / (returns.select(pl.col(f"{symbol}_std_dev")).item() + 1e-9)
                              for symbol in symbols}

        if any(weight <= 0 for weight in vol_scaled_weights.values()):
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(vol_scaled_weights.values())
        normalized_weights = {symbol: weight / total_weight for symbol, weight in vol_scaled_weights.items()}

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: max(normalized_weights[symbol], 1e-9) for symbol in symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest