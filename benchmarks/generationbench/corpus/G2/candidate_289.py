from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when asset prices eventually return to the long-term average. "
        "Short-horizon mean reversion strategies look for extreme price deviations from this "
        "average and bet on a reversal."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        ).to_series().to_list()[0]

        signals: dict[str, float] = {}
        for symbol in symbols:
            symbol_history = history.select(pl.col("symbol"), "session_date", "adj_close")
            close_values = [float(v) for v in symbol_history.filter(
                (pl.col("symbol") == pl.lit(symbol)) & (pl.col("adj_close").is_not_null())
            )["adj_close"].to_list()]

            if len(close_values) < self._window + 1:
                continue

            last_close = close_values[-1]
            deviation = (last_close - mean_close[symbols.index(symbol)]) / mean_close[symbols.index(symbol)]
            if abs(deviation) > 0.2:  # Example threshold for reversion
                signals[symbol] = 1.0 / len(symbols)

        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol, weight in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest