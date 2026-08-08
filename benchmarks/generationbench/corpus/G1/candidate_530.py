from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Trend-following strategies aim to capture the momentum of price movements. By scaling "
        "trading decisions with volatility, we can adjust our positions based on market conditions."
    )

    def __init__(self, window: int = 20, factor: float = 1.5) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        closes = view.closes(lookback=self._window)

        # Calculate log returns
        df_returns = (
            history.lazy()
            .select(pl.col("symbol"), pl.col("session_date"), (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
            .collect()
        )

        # Calculate rolling volatility for each symbol
        volatilities = (
            df_returns.group_by("symbol")
            .agg((pl.col("return").abs().mean() * self._factor).alias("volatility"))
            .select(pl.col("symbol"), "volatility")
        ).to_pandas()

        # Determine long and short symbols based on volatility
        buys = volatilities[volatilities["volatility"] > 0]["symbol"].tolist()
        sells = volatilities[volatilities["volatility"] < 0]["symbol"].tolist()

        weights: dict[str, float] = {}
        for symbol in symbols:
            if symbol in buys and symbol not in closes.columns:
                continue
            elif symbol in sells and symbol not in closes.columns:
                continue
            weight = 1.0 / len(buys) if symbol in buys else -1.0 / len(sells)
            weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest