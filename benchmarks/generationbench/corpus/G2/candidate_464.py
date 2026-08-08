from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong market sentiment. "
        "A significant increase in volume while the price is moving in a particular direction can "
        "signal a continuation or acceleration of the trend. This strategy aims to capitalize on such"
        "moments by entering trades that align with recent price movements and volume surges."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        moves: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            recent_closes = [float(v) for v in history[symbol].drop_nulls().to_list()[-self._window:]]
            recent_volumes = [int(v) for v in history.select(pl.col(symbol).shift(-1)).column(0).drop_nulls().to_list()[-self._window:]]

            if len(recent_closes) < self._window or len(recent_volumes) < self._window:
                continue

            # Calculate the directional move
            move = (recent_closes[-1] - recent_closes[0]) / abs(recent_closes[0])
            # Check for a significant volume increase in the last period
            if max(recent_volumes) > 1.5 * sum(recent_volumes) / len(recent_volumes):
                moves[symbol] = move

        picks: list[str] = [symbol for symbol, _ in sorted(moves.items(), key=lambda item: abs(item[1]), reverse=True)[:3]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest