from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify and exploit trending behavior in the market. "
        "During periods of high volatility, assets that have been rising tend to continue rising, and vice versa."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            (history["close"] - history["open"])
            .abs()
            .mean()
            / history["close"]
            .mean()
        )
        trend = (history["close"].max() - history["close"].min()) / history["close"].mean()

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            close_price = view.latest_close()[symbol]
            recent_trend = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date", descending=True)
                .select(
                    (pl.col("close").max() - pl.col("close").min()) / pl.col("close").mean()
                )
                .head(1)
                .to_dict(True)[0][0]
            )

            if volatility[symbol] > self._threshold * trend:
                signals[symbol] = 1.0
            else:
                signals[symbol] = 0.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest