from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies aim to capture trends by smoothing price "
        "changes and scaling them with historical volatility. Higher volatility periods can lead "
        "to more aggressive trend following, while lower volatility periods suggest caution or "
        "mean reversion."
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
            .sort("session_date", descending=False)
            .with_column(pl.col("r").rank(method="ordinal", descending=True).alias("rank"))
        )

        # Calculate historical volatility
        vol = (
            history.groupby("symbol").agg(
                (pl.col("r").std().over(pl.all()).alias("volatility"))
            )
        ).height

        if vol < 1:
            return Signal(information_available_at=stamp, weights={})

        scaled_trends: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in history.columns or "r" not in history.columns:
                continue
            returns = [float(v) for v in history[history["symbol"] == symbol]["r"].to_list()]
            if len(returns) < self._window:
                continue
            rank = float(history.filter(pl.col("symbol") == symbol)["rank"].item())
            weight = 1.0 / (vol + 1) * rank

            scaled_trends.append((symbol, weight))

        weights = {s: w for s, w in sorted(scaled_trends, key=lambda x: -x[1])}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest