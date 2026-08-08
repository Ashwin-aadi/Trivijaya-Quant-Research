from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical observation that low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "By tilting the portfolio towards lower-risk assets based on their 12-month rolling standard deviation of returns, we aim to capture excess returns while maintaining market exposure and diversification benefits."
    )

    def __init__(self, window: int = 12, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date", descending=True)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        prices = history.select(
            pl.col("symbol").cast(pl.Utf8),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
        )

        # Calculate the rolling standard deviation of returns
        volatilities = (
            prices.groupby("symbol")
            .agg(
                (pl.col("r").std().over(pl.arange(1, self._window + 1)).alias(f"volatility"))
            )
            .select(["symbol", "volatility"])
        )

        # Rank the symbols by their volatility
        ranked_symbols = volatilities.sort("volatility").limit(self._top_n)

        weights: dict[str, float] = {}
        weight_per_symbol = 1.0 / len(ranked_symbols)
        for symbol in ranked_symbols["symbol"]:
            weights[symbol] = weight_per_symbol

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest