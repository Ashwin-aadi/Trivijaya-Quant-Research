from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and returns are tendency to revert "
        "to a long-term mean. In the short term (20 days), price deviations from the mean can "
        "create trading opportunities. If an asset's price has fallen significantly below its "
        "mean, it is likely to bounce back towards the mean."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.select(
            pl.col(pl.Utf8).exclude("session_date").mean().alias("mean")
        ).select("mean").to_series()[0]

        symbols_to_trade: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            if abs(latest_close - mean_close) > 1.5 * (latest_close - closes[symbol].mean().item()):
                symbols_to_trade.append(symbol)

        weights: dict[str, float] = {s: 1.0 / len(symbols_to_trade) for s in symbols_to_trade}
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