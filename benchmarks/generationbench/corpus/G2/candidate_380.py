from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and are often "
        "followed by continued price movement in the same direction. By identifying such moves, "
        "we can exploit this momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = _find_volume_confirmed_moves(history)
        weight = 1.0 / len(top_n_symbols) if top_n_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_volume_confirmed_moves(history: pl.DataFrame) -> list[str]:
    symbols = [symbol for symbol in history.columns if symbol != "session_date"]
    top_n_symbols = []
    for symbol in symbols:
        price_changes = (history[symbol].to_list()[1:] - history[symbol].to_list()[:-1])
        volume_changes = history[f"{symbol}_volume"].to_list()[1:]
        
        directional_changes = [price_change * vol_change > 0 for price_change, vol_change in zip(price_changes, volume_changes)]
        
        if all(directional_changes):
            top_n_symbols.append(symbol)
    
    return top_n_symbols[:5]