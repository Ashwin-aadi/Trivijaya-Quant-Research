from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting position sizes "
        "based on recent volatility. High volatility periods reduce exposure to prevent large losses, "
        "while low volatility allows for larger positions as risk is perceived to be lower."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].unique().to_list()]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.select(pl.col("adj_close").mean())
            .to_numpy()[0][0]
        )

        vol = (history.select(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).select((pl.col("returns").std() * pl.lit(self._vol_window)).alias("volatility"))
               .to_numpy()[0][0]
        )

        weights = {symbol: (max(0, mean_close / close - 2 * vol) / len(symbols)) for symbol in symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest