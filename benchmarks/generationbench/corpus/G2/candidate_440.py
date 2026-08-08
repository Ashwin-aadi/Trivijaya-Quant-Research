from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate a shift in investor sentiment towards "
        "a stock. If a stock has a significant upward move on high volume, it often suggests that"
        " new buying interest is driving the price up, which may lead to further gains."
    )

    def __init__(self, window: int = 5, threshold_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._threshold_volume_ratio = threshold_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data: dict[str, list[float]] = {}
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            open_prices = [float(o) for o in df["open"].to_list()]
            close_prices = [float(c) for c in df["close"].to_list()]
            volumes = [float(v) for v in df["volume"].to_list()]

            if len(open_prices) < self._window:
                continue

            latest_close_price = float(df.filter(pl.col("session_date") == stamp).select("adj_close").row(0)[0])
            last_window_close_price = close_prices[-1]

            # Check for significant move
            price_move_ratio = (latest_close_price - last_window_close_price) / abs(last_window_close_price)
            if not 0.05 < price_move_ratio < 0.2:
                continue

            # Check for high volume confirmation
            latest_volume = volumes[-1]
            average_volume = sum(volumes[:-1]) / (self._window - 1)
            volume_ratio = latest_volume / average_volume

            if not self._threshold_volume_ratio > volume_ratio:
                continue

            symbol_data[symbol] = [price_move_ratio, volume_ratio]

        if not symbol_data:
            return Signal(information_available_at=stamp, weights={})

        # Rank symbols by combined score
        scores = [
            (symbol, price_move * 0.5 + volume_ratio * 0.5)
            for symbol, (price_move, volume_ratio) in symbol_data.items()
        ]
        ranked_symbols = sorted(scores, key=lambda x: -x[1])

        top_symbols = [s for s, _ in ranked_symbols[:3]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest