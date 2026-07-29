from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have more stable returns and are less affected by market "
        "volatility. Tilting the portfolio towards low-volatility stocks can reduce overall risk."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty() or len(history["symbol"].unique()) < 10:
            return Signal(information_available_at=stamp, weights={})

        volatilities = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close").std().alias("volatility"))
            )
            .sort("volatility", descending=False)
            .select("symbol", "volatility")
            .to_dict(as_series=False)
        )

        symbols = list(volatilities["symbol"][:10])
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights=dict(zip(symbols, [weight] * len(symbols)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest