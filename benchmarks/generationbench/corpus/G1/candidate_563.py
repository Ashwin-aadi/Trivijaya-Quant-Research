from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakoutAndMeanReversion(Strategy):
    rationale = (
        "This strategy identifies stocks with strong volume breakout and mean reversion "
        "tendencies. Stocks that have a significant volume increase followed by a price "
        "recovery are often seen as oversold and due for a bounce."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        volume_breakouts: list[str] = []
        mean_reversions: list[str] = []

        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in closes.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            vol_values = [int(v) for v in history[f"{symbol}_volume"].to_list()]

            # Check for volume breakout
            if (
                vol_values[-1] > max(vol_values)
                and (close_values[-1] - close_values[0]) / close_values[0] < 0.1
            ):
                volume_breakouts.append(symbol)

            # Check for mean reversion
            recent_close = closes[symbol].to_list()[-self._window :]
            if len(recent_close) < self._window:
                continue
            avg_price = sum(float(v) for v in recent_close) / self._window
            if abs(close_values[0] - avg_price) / avg_price > 0.1:
                mean_reversions.append(symbol)

        volume_breakouts = list(set(volume_breakouts))
        mean_reversions = list(set(mean_reversions))

        picks = volume_breakouts + mean_reversions
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.5 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in picks[: self._top_n]
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest