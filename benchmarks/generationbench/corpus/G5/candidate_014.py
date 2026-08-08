from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFT(Strategy):
    rationale = (
        "This strategy follows trends by identifying stocks with high recent volatility. "
        "High volatility suggests a change in trend or increased uncertainty, which can be an entry point."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .drop_nulls()
            .select(["symbol", "session_date", "r"])
        )

        # Calculate volatility
        volatilities = (
            history.groupby("symbol")
            .agg(
                pl.col("r").std().alias("volatility"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("mean_return"),
            )
            .select(["symbol", "volatility"])
        )

        # Filter for symbols with high volatility
        top_symbols = volatilities.sort(by="volatility", descending=True)["symbol"].to_list()[: self._top_n]

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