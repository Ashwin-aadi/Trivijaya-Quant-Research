from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Directional moves in the market are often more significant and durable when they "
        "are accompanied by increased trading volume. This strategy identifies such moves to "
        "capitalize on potential momentum."
    )

    def __init__(self, window: int = 10, threshold: float = 1.2) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).sort(
                "session_date"
            )
            open_prices = [float(v) for v in symbol_history["open"].to_list()]
            close_prices = [float(v) for v in symbol_history["close"].drop_nulls().to_list()]
            volumes = [float(v) for v in symbol_history["volume"].to_list()]

            if len(close_prices) < self._window:
                continue

            latest_close = close_prices[-1]
            last_window_close = close_prices[-self._window :]
            max_movement = max([abs(x - y) for x, y in zip(last_window_close[:-1], last_window_close[1:])])

            if (
                abs(latest_close - min(last_window_close)) >= self._threshold * max_movement
                or abs(latest_close - max(last_window_close)) >= self._threshold * max_movement
            ):
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest