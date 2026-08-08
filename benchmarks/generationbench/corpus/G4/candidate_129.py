from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over time due "
        "to risk aversion and long-term risk-return trade-offs. This strategy aims to "
        "construct a portfolio of low-volatility stocks, rebalancing periodically to "
        "capitalize on this effect."
    )

    def __init__(self, window: int = 20, portfolio_size: int = 30) -> None:
        self._window = window
        self._portfolio_size = portfolio_size

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day realized volatility
        history_with_volatility = (
            history.with_columns(
                (pl.col("high") - pl.col("low")).alias("range"),
                ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("return"),
            )
            .with_column(
                (pl.col("range").std() * (252**0.5) / 20).alias("volatility")
            )
            .sort("session_date", descending=True)
        )

        # Get the most recent volatility for each stock
        latest_closes = view.closes(lookback=self._window)
        volatilities = (
            history_with_volatility.groupby("symbol").agg(
                pl.col("volatility").first().alias("latest_volatility")
            )
            .join(latest_closes, on="symbol", how="inner")
            .select("symbol", "latest_volatility")
        )

        # Rank stocks by volatility
        ranked_stocks = (
            volatilities.sort("latest_volatility").rows()
            if not volatilities.is_empty()
            else pl.DataFrame({"symbol": [], "latest_volatility": []})
        )

        if len(ranked_stocks) < self._portfolio_size:
            return Signal(information_available_at=stamp, weights={})

        # Select the least volatile stocks
        selected_symbols = [r[0] for r in ranked_stocks[: self._portfolio_size]]

        # Allocate weight to each selected stock
        weight_per_symbol = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest