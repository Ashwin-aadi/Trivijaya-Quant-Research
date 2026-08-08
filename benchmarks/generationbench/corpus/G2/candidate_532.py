from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price levels revert to a long-term mean. This strategy exploits this tendency by "
        "buying stocks that have fallen below their trailing moving average and selling those "
        "that have risen above it."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=self._window)
        symbols = [symbol for symbol in view.symbols if symbol in closes.columns]

        def trailing_average(symbol: str) -> float:
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            )
            avg_close = (
                symbol_history.sort("session_date").select(
                    (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("r")
                ).sort("session_date").head(self._window)["r"].sum()
            ) / self._window
            return avg_close

        weights: dict[str, float] = {}
        for symbol in symbols:
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < 2:
                continue
            trailing_avg = trailing_average(symbol)
            current_price = recent_closes[-1]
            reversion_signal = (current_price - trailing_avg) / trailing_avg

            if abs(reversion_signal) > self._threshold:
                weights[symbol] = 1.0

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest