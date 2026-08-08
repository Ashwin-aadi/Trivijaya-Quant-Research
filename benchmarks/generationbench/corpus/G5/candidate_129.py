from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "By tilting our portfolio towards low volatility, we aim to capture this anomaly."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[dict[str, float]] = []
        for symbol in symbols:
            prices = history.filter(pl.col("symbol") == symbol)[
                ["session_date", "adj_close"]
            ]
            returns = [float(prices.select("adj_close").to_series().pct_change().drop_nulls().sum())]
            if len(returns) < 1:
                continue
            vol = (sum(r**2 for r in returns) / len(returns)).sqrt()
            volatilities.append({"symbol": symbol, "volatility": float(vol)})

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        sorted_volatilities = sorted(volatilities, key=lambda x: x["volatility"])
        picks = [sv["symbol"] for sv in sorted_volatilities[:10]]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest