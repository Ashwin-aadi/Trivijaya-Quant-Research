from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "A stock with a higher recent relative strength compared to the NIFTY 100 index "
        "suggests it is outperforming its peers and may be a good candidate for entry."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_index = view.latest_close()["^NIFTY 100"] or 0.0
        strength_ratios: list[float] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol == "^NIFTY 100":
                continue
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue
            recent_close = close_values[-1]
            index_close = market_index * (close_values / market_index)[-1]
            strength_ratio = recent_close / index_close - 1.0
            strength_ratios.append(strength_ratio)

        top_n_strength_symbols = sorted(zip(view.symbols, strength_ratios), key=lambda x: x[1], reverse=True)[:5]
        if not top_n_strength_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_strength_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_n_strength_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest