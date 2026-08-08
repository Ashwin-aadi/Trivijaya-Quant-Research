from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks with the highest relative strength against the NIFTY 100 index. "
        "Relative strength is measured by comparing the stock's recent price performance to that of the broader market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_close = view.latest_close()["NIFTY 100"]
        symbol_strengths: list[tuple[str, float]] = []
        
        for symbol in view.symbols:
            if symbol == "NIFTY 100":
                continue
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            
            nifty_values = [float(nifty_close.get(k).with_default(0.0)) for k, v in values.items()]
            strength = (values[-1] - values[0]) / (nifty_values[-1] - nifty_values[0])
            symbol_strengths.append((symbol, strength))

        symbol_strengths.sort(key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in symbol_strengths[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest