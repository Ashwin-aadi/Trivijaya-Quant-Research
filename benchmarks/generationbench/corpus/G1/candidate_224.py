from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A directional move in price that is confirmed by increased volume suggests "
        "stronger market sentiment and is more likely to continue."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.is_empty() or history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_low_ratio = (
                (history[f"{symbol}_high"] / history[f"{symbol}_low"])
                .to_list()[-self._window:]
            )
            volume_ratio = (
                (history[f"{symbol}_volume"] / history[f"{symbol}_volume"].shift(1))
                .to_list()[1:-1]
            )

            if all(high_low_ratio[i] > 1.01 for i in range(len(high_low_ratio))):
                if any(volume_ratio[i] > 1.05 for i in range(len(volume_ratio))):
                    picks.append(symbol)

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