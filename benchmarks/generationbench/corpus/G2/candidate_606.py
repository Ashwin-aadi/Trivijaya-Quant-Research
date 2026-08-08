from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that outperform the broader market in the short term are likely to continue "
        "outperforming due to momentum effects. This strategy identifies and invests in the top"
        "performers relative to the NIFTY 100 index."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        index_close = float(view.latest_close()["^NIFTY 100"])
        relative_strength: list[float] = []
        for symbol in view.symbols:
            if symbol == "^NIFTY 100":
                continue
            symbol_closes = [float(v) for v in closes[symbol].to_list()]
            if len(symbol_closes) < self._window or index_close <= 0.0:
                continue
            ratio = max(symbol_closes[-1] / index_close - 1.0, 0.0)
            relative_strength.append(ratio)

        top_n_indices = sorted(range(len(relative_strength)), key=lambda i: relative_strength[i], reverse=True)[: self._top_n]
        weights = {view.symbols[i]: 1.0 / len(top_n_indices) for i in top_n_indices}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest