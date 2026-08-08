from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "This strategy capitalizes on the tendency for stock prices to revert to historical "
        "levels after short-term deviations. By identifying stocks that have fallen or risen"
        " significantly from their 12-month average price, we can generate trading signals."
    )

    def __init__(self, window: int = 365, threshold: float = 0.05, top_n: int = 20) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_prices = history.groupby("symbol").agg(
            (pl.col("close") / pl.col("close").shift(self._window - 1) - 1.0).mean().alias("avg_return")
        )
        avg_prices = avg_prices.with_columns(
            (pl.col("close") * (1 + pl.col("avg_return"))).alias("trailing_avg_price")
        )

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in avg_prices.columns or symbol not in history.columns:
                continue
            current_close = float(view.latest_close()[symbol])
            trailing_avg = float(avg_prices[avg_prices["symbol"] == symbol]["trailing_avg_price"])
            price_ratio = current_close / trailing_avg

            if 1 - self._threshold > price_ratio >= 0.95:
                signals[symbol] = 1.0
            elif 1 + self._threshold < price_ratio <= 1.05:
                signals[symbol] = -1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        picks = [s for s, w in sorted_signals[: self._top_n]]
        weight = 1.0 / len(picks) if picks else 0
        return Signal(information_available_at=stamp, weights={s: weight for s in picks})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest