from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMoves(Strategy):
    rationale = (
        "Volume-weighted average price (VWAP) over a 5-day lookback period can indicate "
        "volume-confirmed directional moves in the market. By holding up to three equities "
        "with predefined weights, we aim to capture these moves more effectively."
    )

    def __init__(self, window: int = 5, max_positions: int = 3) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        vwaps: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].to_list()]
            volumes = [float(v) for v in history[f"v_{symbol}"].to_list()]
            vwaps[symbol] = sum(c * v for c, v in zip(adj_closes, volumes)) / sum(volumes)

        sorted_vwaps = sorted(vwaps.items(), key=lambda x: x[1], reverse=True)
        picks: list[str] = [symbol for symbol, _ in sorted_vwaps[:self._max_positions]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight_1 = 0.4
        weight_2 = 0.35
        weight_3 = 0.25

        weights = {picks[0]: weight_1}
        if len(picks) > 1:
            weights[picks[1]] = weight_2
        if len(picks) > 2:
            weights[picks[2]] = weight_3

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