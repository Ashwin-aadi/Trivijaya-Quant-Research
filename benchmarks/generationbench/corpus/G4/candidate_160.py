from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "This strategy exploits price-level reversion by identifying stocks that have deviated significantly from "
        "their 50-day simple moving average (SMA). Overbought and oversold conditions are expected to revert towards a "
        "mean level due to market forces and investor behavior. By shorting overpriced stocks and buying undervalued ones, "
        "we aim to profit from temporary price fluctuations around historical levels."
    )

    def __init__(self, window: int = 50, threshold_high: float = 2.0, threshold_low: float = -2.0) -> None:
        self._window = window
        self._threshold_high = threshold_high / 100
        self._threshold_low = threshold_low / 100

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        sma_col = f"sma_{self._window}"
        recent_closes = view.closes(lookback=self._window)

        # Compute 50-day SMA for each stock
        history = (
            history.group_by("symbol")
                   .with_column(
                       (pl.col("adj_close").rolling_mean(self._window)).alias(sma_col)
                   )
        )

        # Calculate deviation from 50-day SMA
        recent_closes = recent_closes.with_column(pl.col("session_date").to_frame())
        recent_closes = (
            recent_closes.join(history, on="symbol", how="left")
                         .with_column(
                             (pl.col("adj_close") - pl.col(sma_col)).alias("deviation")
                         )
        )

        # Set thresholds and rank candidates
        buys: list[str] = []
        shorts: list[str] = []
        for symbol in symbols:
            recent_vals = [float(v) for v in recent_closes[recent_closes["symbol"] == symbol]["deviation"].to_list()]
            if len(recent_vals) < self._window + 1:
                continue
            high_threshold_reached = any(d >= self._threshold_high for d in recent_vals)
            low_threshold_reached = any(d <= self._threshold_low for d in recent_vals)
            if high_threshold_reached:
                shorts.append(symbol)
            elif low_threshold_reached:
                buys.append(symbol)

        weight = 1.0 / max(len(buys), len(shorts))
        return Signal(
            information_available_at=stamp, weights={**{s: -weight for s in shorts}, **{s: weight for s in buys}}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest