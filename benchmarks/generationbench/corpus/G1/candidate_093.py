from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Historically, low-volatility stocks have shown to have a lower dispersion in returns "
        "and can provide more stable performance. Tilting the portfolio towards these stocks "
        "can potentially enhance risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history.group_by("symbol")
            .agg(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).mean().alias("avg_return"),
                (pl.col("close") / pl.col("close").shift(1) - 1.0).std().alias("volatility"),
            )
            .sort("volatility", descending=False)
            .head(self._window)["symbol"]
            .to_list()
        )

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(volatilities)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in volatilities}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest