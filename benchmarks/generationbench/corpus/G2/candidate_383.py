from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeVolatilityMomentum(Strategy):
    rationale = (
        "This strategy aims to identify stocks that show high volatility followed by strong momentum. "
        "High volatility can indicate significant price movement and risk, while a subsequent positive momentum suggests the potential for profit."
    )

    def __init__(self, vol_window: int = 20, mom_window: int = 10) -> None:
        self._vol_window = vol_window
        self._mom_window = mom_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._vol_window + self._mom_window)

        if closes.height < self._vol_window + self._mom_window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._vol_window + self._mom_window:
                continue

            vol = (adj_closes[-self._vol_window:] - adj_closes[:-self._vol_window]).std()
            mom = (adj_closes[-1] / adj_closes[-(self._mom_window + 1)] - 1.0)
            if vol > 0 and mom > 0:
                volatilities.append(symbol)

        volatilities = volatilities[:5]
        if not volatilities:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(volatilities)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in volatilities}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest