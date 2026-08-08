from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "Combining volatility and momentum can capture varying market conditions. Volatility "
        "signals potential reversals or mean reversion opportunities, while momentum signals a "
        " continuation of the current trend. This composite strategy uses both to filter out noisy "
        "periods."
    )

    def __init__(self, vol_window: int = 10, mom_window: int = 20) -> None:
        self._vol_window = vol_window
        self._mom_window = mom_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._vol_window + self._mom_window)

        if closes.height < self._vol_window + self._mom_window:
            return Signal(information_available_at=stamp, weights={})

        vol_signals: dict[str, float] = {}
        mom_signals: dict[str, float] = {}

        for symbol in view.symbols:
            adj_closes = [float(v) for v in closes[symbol].to_list()]
            if len(adj_closes) < self._vol_window + self._mom_window:
                continue

            vol_mean = pl.Series(adj_closes[-self._vol_window:]).mean()
            mom_return = (adj_closes[-1] - adj_closes[-self._mom_window]) / adj_closes[
                -self._mom_window
            ]
            if len(adj_closes) < self._vol_window + 1:
                vol_signals[symbol] = 0.5
                mom_signals[symbol] = 0.5
            else:
                vol_signals[symbol] = (adj_closes[-1] > vol_mean) * 1.0
                mom_signals[symbol] = (mom_return >= 0) * 1.0

        combined_signal = {
            symbol: 2 * vol_signals[symbol] + 3 * mom_signals[symbol]
            for symbol in view.symbols
        }

        top_symbols = sorted(combined_signal.items(), key=lambda x: x[1], reverse=True)
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(top_symbols), 5)
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol, _ in top_symbols[:top_n]
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest