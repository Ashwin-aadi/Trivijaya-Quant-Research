from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify trending stocks by measuring the "
        "volatility of recent price movements and scaling that volatility against a fixed period's "
        "returns. When the current return exceeds this scaled value, it suggests a trend may be forming."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        symbols = view.symbols

        # Calculate daily returns
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=True)
            .tail(self._window + 1)
        )

        # Calculate mean and std of returns over the lookback period
        means = history.select(pl.col("r").mean()).item()
        stds = history.select(pl.col("r").std()).item()

        # Scale the mean return by the standard deviation
        scaled_mean = means * self._scale_factor

        picks: list[str] = []
        for symbol in symbols:
            if symbol not in latest_close or latest_close[symbol] is None:
                continue
            current_return = (latest_close[symbol] / history.select(pl.col(symbol).first()).item() - 1.0)
            if current_return > scaled_mean:
                picks.append(symbol)

        # Handle cases where no symbols meet the criteria
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return view.as_of
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest