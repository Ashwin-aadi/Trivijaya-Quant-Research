from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves can indicate a strong underlying trend. "
        "When there is a significant increase in volume alongside a price move, it often "
        "suggests that the market is consolidating and preparing for further movement."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        moves_and_volumes = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            if df.height < 2:
                continue
            latest_close = float(df.select("adj_close").tail(1)["adj_close"][0])
            previous_close = float(df.select("adj_close").tail(2)["adj_close"][1])
            volume_change = (float(df.select("volume").tail(1)["volume"][0]) - 
                             float(df.select("volume").tail(2)["volume"][1]))
            if latest_close > previous_close and volume_change > 0:
                moves_and_volumes.append((symbol, "up", volume_change))
            elif latest_close < previous_close and volume_change < 0:
                moves_and_volumes.append((symbol, "down", volume_change))

        if not moves_and_volumes:
            return Signal(information_available_at=stamp, weights={})

        top_moves = sorted(moves_and_volumes, key=lambda x: abs(x[2]), reverse=True)[:5]
        weight = 1.0 / len(top_moves)
        selected_symbols = [move[0] for move in top_moves]
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_list()[0]
    assert isinstance(newest, date)
    return newest