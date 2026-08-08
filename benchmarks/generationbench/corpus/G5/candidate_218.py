from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling trades based on "
        "recent volatility. High recent volatility suggests strong market movement and thus "
        "potential for larger gains or losses. This strategy enters long positions in symbols "
        "with high recent volatility."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )
        history = history.sort("session_date")

        # Filter out symbols not present in the full history window
        present_symbols = set(history["symbol"].to_list())
        history = history.filter(pl.col("symbol").is_in(present_symbols))

        # Calculate volatility for each symbol using standard deviation
        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in present_symbols:
                continue
            returns = history.filter(pl.col("symbol") == symbol)["return"].to_list()
            if len(returns) < self._window:
                continue
            volatility = (sum([r**2 for r in returns]) / self._window) ** 0.5
            volatilities[symbol] = volatility

        # Identify top symbols by volatility
        sorted_symbols = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s[0] for s in sorted_symbols[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest