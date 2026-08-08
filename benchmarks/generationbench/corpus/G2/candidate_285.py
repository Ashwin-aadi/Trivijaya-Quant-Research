from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed high-volatility stocks. "
        "This is often attributed to the risk premium that investors demand for holding "
        "riskier assets. By tilting towards low-volatility stocks, one can potentially "
        "reduce overall portfolio volatility and capture this risk premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Compute volatility for each symbol over the lookback window
        volatilities = (
            history.groupby("symbol")
            .agg(
                pl.col("r").std().alias("volatility"),
            )
            .select(["symbol", "volatility"])
        )

        # Sort symbols by their volatilities and select the top N lowest-volatility stocks
        sorted_volatilities = volatilities.sort("volatility")
        picks: list[str] = [row["symbol"] for row in sorted_volatilities.rows()][:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Assign equal weight to each selected stock
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest