from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed high-volatility stocks. "
        "This phenomenon can be attributed to risk-return trade-offs and investor behavior. "
        "By tilting towards low-volatility stocks, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_volatility = (
            history.select(
                pl.col("symbol").alias("SYMBOL"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .with_columns(pl.col("r").abs().alias("vol"))
            .group_by("SYMBOL")
            .agg(
                (pl.col("vol").mean()).alias("avg_vol"),
                (pl.col("adj_close")[-1]).alias("latest_price"),
            )
            .sort("avg_vol", descending=False)
            .select(["SYMBOL", "latest_price"])
        )

        symbols = mean_volatility["SYMBOL"].to_list()[:5]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest