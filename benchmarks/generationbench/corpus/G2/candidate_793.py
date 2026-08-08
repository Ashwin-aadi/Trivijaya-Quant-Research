from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that after a security has experienced an extreme price change, "
        "it is likely to revert towards its historical mean. In the context of Indian equities, "
        "a significant move away from the 20-day moving average can indicate an overbought or oversold condition."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()
        mean_price = (
            pl.concat([history.select(pl.col("symbol")), history.select("adj_close")])
            .group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .to_dict(True)
        )
        recent_prices = [
            (symbol, latest_close[symbol] / mean_price[symbol]["mean"] - 1.0)
            for symbol in view.symbols
            if symbol in latest_close and symbol in mean_price["symbol"]
        ]

        # Filter out symbols that are not significantly deviated from the mean
        picks = [
            (symbol, deviation)
            for symbol, deviation in recent_prices
            if abs(deviation) >= self._threshold
        ]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={symbol: weight for symbol, _ in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest