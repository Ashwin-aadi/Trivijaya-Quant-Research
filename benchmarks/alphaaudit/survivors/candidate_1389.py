from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have lower downside risk and historically offer "
        "better risk-adjusted returns. By tilting our portfolio towards low-volatility "
        "stocks, we aim to capture these benefits."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        grouped_history = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility")),
                pl.col("adj_close").last().alias("latest_price"),
            )
            .sort("volatility", descending=False)
            .select(["symbol", "volatility", "latest_price"])
        )

        if grouped_history.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in grouped_history.head(5).rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest