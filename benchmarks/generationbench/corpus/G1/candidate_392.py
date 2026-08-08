from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves are more likely to be sustained than those "
        "without significant volume support. This strategy aims to capture such moves by "
        "identifying symbols with strong price movements and substantial volume increases."
    )

    def __init__(self, window: int = 10, min_volume_increase: float = 2.0) -> None:
        self._window = window
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = set(history["symbol"].to_list())
        signals: dict[str, float] = {}

        for symbol in symbols:
            sym_history = history.filter(pl.col("symbol") == symbol)
            if sym_history.height < self._window + 1:
                continue

            open_prices = [float(v) for v in sym_history["open"].to_list()]
            close_prices = [float(v) for v in sym_history["close"].to_list()]

            # Calculate daily returns
            returns = [(close - open_) / open_ if open_ != 0 else float('inf') for open_, close in zip(open_prices[:-1], close_prices[1:])]

            # Check for significant price movement and volume increase
            last_session_open, last_session_close = open_prices[-2], close_prices[-1]
            last_session_volume = float(sym_history.filter(pl.col("symbol") == symbol)["volume"].tail(1).to_list()[0])
            prev_session_volume = float(history.filter((pl.col("symbol") == symbol) & (pl.col("session_date") < view.as_of)).sort("session_date", descending=True).filter(pl.col("volume").is_not_null()).head(1)['volume'])

            if last_session_close / last_session_open > 0.95 and last_session_volume / prev_session_volume >= self._min_volume_increase:
                signals[symbol] = 1.0

        return Signal(information_available_at=stamp, weights={s: weight for s, weight in signals.items() if weight != 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest