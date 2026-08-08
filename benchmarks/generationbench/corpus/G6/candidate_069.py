from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy aims to capture returns from stocks with lower volatility while "
        "maintaining market exposure through active management. It uses the standard deviation "
        "of daily returns over the last 60 days for selection and rebalances monthly to ensure "
        "alignment with low-volatility criteria."
    )

    def __init__(self, window: int = 60, top_percentile: float = 0.4, stop_loss: float = 0.1) -> None:
        self._window = window
        self._top_percentile = top_percentile
        self._stop_loss = stop_loss

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        returns_df = history.with_columns(daily_returns)
        volatilities = returns_df.group_by("symbol").agg(
            pl.col("r").std().alias("volatility")
        ).sort("volatility")

        if volatilities.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        cutoff = int(volatilities.height * (1 - self._top_percentile))
        low_vol_stocks = volatilities["symbol"][cutoff:].to_list()
        weight = 1.5 / len(low_vol_stocks) if low_vol_stocks else 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_vol_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest