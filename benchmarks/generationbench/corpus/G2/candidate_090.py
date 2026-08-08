from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed the broad market over a recent period are more likely "
        "to continue outperforming due to momentum effects. This strategy buys assets with the "
        "highest relative strength."
    )

    def __init__(self, lookback: int = 30) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()["^NIFTY 100"]/view.latest_close()["^NIFTY 100"]
        relative_strength: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol == "^NIFTY 100":
                continue
            symbol_close = closes[symbol].to_list()
            market_close_values = [float(v) for v in market_close.to_list()]
            if len(symbol_close) < self._lookback or len(market_close_values) < self._lookback:
                continue

            avg_symbol_return = sum((symbol_close[i] / symbol_close[i - 1] - 1.0
                                     for i in range(1, self._lookback))) / (self._lookback - 1)
            avg_market_return = sum((market_close_values[i] / market_close_values[i - 1] - 1.0
                                     for i in range(1, self._lookback))) / (self._lookback - 1)

            if avg_symbol_return > avg_market_return:
                relative_strength[symbol] = avg_symbol_return / avg_market_return

        sorted_symbols = [k for k, v in sorted(relative_strength.items(), key=lambda item: item[1], reverse=True)]
        top_n = min(self._lookback // 3, len(sorted_symbols))  # Buy the top third of symbols
        weight = 1.0 / top_n
        return Signal(information_available_at=stamp, weights={s: weight for s in sorted_symbols[:top_n]})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest