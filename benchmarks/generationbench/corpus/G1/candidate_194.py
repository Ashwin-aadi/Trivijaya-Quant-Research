from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the relationship between price movement "
        "and volatility. When a stock's recent volatility is low, it suggests that future "
        "returns are likely to be higher. This strategy captures these periods of reduced "
        "volatility for potential profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close") - pl.col("adj_close").shift(1)).abs().mean()
            )
            .with_columns(pl.col("adj_close").std().alias("volatility"))
            .with_columns(
                ((pl.col("adj_close").std() / pl.col("adj_close").mean()) * 100).alias(
                    "volatility_scaled"
                )
            )
        )

        symbols = volatility.select("symbol").to_dict(True)["symbol"]
        picks: list[str] = []
        for symbol in symbols:
            row = volatility.filter(pl.col("symbol") == symbol).rows()[0]
            if float(row["volatility"]) / float(row["adj_close"].mean()) < self._threshold:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest