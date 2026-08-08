from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and confidence. "
        "Highly liquid stocks are likely to have more balanced demand and supply, reducing idiosyncratic risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "volume" not in history.columns:
                continue
            volume_values = [float(v) for v in history[f"{symbol}.volume"].to_list()]
            mean_volume = sum(volume_values[-self._window:]) / len(volume_values[-self._window:])
            liquidity_scores[symbol] = mean_volume

        sorted_symbols = [
            symbol for _, symbol in sorted(liquidity_scores.items(), key=lambda item: -item[1])
        ][: min(len(view.symbols), self._window)]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest