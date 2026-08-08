from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting assigns weights to stocks based on their "
        "daily trading volume. This approach ensures that more liquid stocks are given "
        "greater weight in the portfolio, aiming to minimize execution costs and impact."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity = history.select(pl.col("symbol"), pl.col("volume").sum().alias("total_volume"))
        total_liquidity = float(liquidity["total_volume"].sum())

        if total_liquidity == 0.0:
            return Signal(information_available_at=stamp, weights={})

        weights: dict[str, float] = {
            symbol: volume / total_liquidity for symbol, volume in liquidity.to_dicts()
        }

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest