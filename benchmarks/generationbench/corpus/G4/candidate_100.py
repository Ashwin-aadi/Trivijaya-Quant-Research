from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionStrategy(Strategy):
    rationale = (
        "This strategy aims to exploit mean reversion in stock prices relative to their historical "
        "price levels. It identifies stocks that have deviated significantly from a trailing moving average "
        "and takes trades when prices cross above or below this reference level."
    )

    def __init__(self, window_short: int = 50, window_long: int = 200) -> None:
        self._window_short = window_short
        self._window_long = window_long

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_long + max(self._window_short, 1))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            close_data = history.select(pl.col("symbol"), pl.col("close")).filter(
                (pl.col("symbol") == symbol)
            )
            sma_50 = close_data["close"].mean().over(window=self._window_short)
            sma_200 = close_data["close"].mean().over(window=self._window_long)
            returns = (
                close_data["close"]
                .shift(-1)
                .rolling_window(1, 1)
                .sum()
                / close_data["close"]
                - 1.0
            )
            std_dev = returns.std().over(window=self._window_short)

            upper_band = sma_50 + 2 * std_dev
            lower_band = sma_50 - 2 * std_dev

            recent_close = view.latest_close()[symbol]

            if (recent_close > upper_band[-1]) and (
                recent_close > max(history.select(pl.col("close")).to_list())
            ):
                picks[symbol] = -1.0
            elif (recent_close < lower_band[-1]) and (
                recent_close < min(history.select(pl.col("close")).to_list())
            ):
                picks[symbol] = 1.0

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in picks.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest