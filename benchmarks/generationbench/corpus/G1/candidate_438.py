from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks with a higher relative strength compared to the broader market "
        "can provide an edge in equity selection. This strategy focuses on outperforming "
        "the overall market as measured by the average return of NIFTY 100 constituents."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty_returns = (closes["close"] / closes["close"].shift(1) - 1.0).to_list()
        symbol_returns = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = (values[-1] / values[0] - 1.0)
            symbol_returns[symbol] = returns

        avg_market_return = sum(nifty_returns) / len(view.symbols)

        picks: list[str] = []
        for symbol, return_val in symbol_returns.items():
            if return_val > avg_market_return:
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