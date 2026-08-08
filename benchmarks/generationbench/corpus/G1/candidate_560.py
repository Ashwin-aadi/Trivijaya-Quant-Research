from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Directional moves in the market are more reliable when they are confirmed by significant volume. "
        "This strategy aims to identify such moves and capitalize on them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]

        signals: dict[str, float] = {}
        for symbol in symbols:
            ohlc = history[[f"{symbol}_open", f"{symbol}_high", f"{symbol}_low", f"{symbol}_close", f"{symbol}_volume"]]
            last_close = float(ohlc["close"].last())
            prev_close = float(ohlc["close"].shift().last())

            if (last_close - prev_close) > 0:
                # Bullish move
                vol_ratio = ohlc[f"{symbol}_volume"].sum() / history[f"{symbol}_volume"].mean()
                if vol_ratio > 1.5:
                    signals[symbol] = 0.2
            elif (prev_close - last_close) > 0:
                # Bearish move
                vol_ratio = ohlc[f"{symbol}_volume"].sum() / history[f"{symbol}_volume"].mean()
                if vol_ratio > 1.5:
                    signals[symbol] = -0.2

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        normalized_weights = {symbol: weight / total_weight for symbol, weight in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=normalized_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest