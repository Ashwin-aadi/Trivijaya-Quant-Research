from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate a strong market sentiment and a "
        "potential for continuation of the trend. High volume on a price move suggests "
        "real buying or selling pressure."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = history.to_dict(False)
        top_moves = {}
        for symbol, data in symbol_prices.items():
            if data["volume"].to_list()[-1] > max(data["volume"].to_list()) * 0.8:
                move = (data["close"] / data["open"] - 1).sum()
                if abs(move) > 0.05:  # Filter out small moves
                    top_moves[symbol] = move

        top_symbols = sorted(top_moves, key=top_moves.get, reverse=True)[:3]
        weights = {symbol: 1 / len(top_symbols) for symbol in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest