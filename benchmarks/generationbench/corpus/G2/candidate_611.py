from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentum(Strategy):
    rationale = (
        "This strategy aims to capture momentum by combining two signals: a long-term "
        "momentum signal based on a 60-day return and a short-term momentum signal based on "
        "a 20-day return. The idea is that securities with strong positive returns over both "
        "timeframes are likely to continue outperforming the market."
    )

    def __init__(self, long_window: int = 60, short_window: int = 20) -> None:
        self._long_window = long_window
        self._short_window = short_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._long_window)
        if closes.height < self._long_window:
            return Signal(information_available_at=stamp, weights={})

        long_mom_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            returns = [float(v) for v in (closes[symbol].drop_nulls().to_list()[1:] / closes[symbol].drop_nulls().to_list()[:-1])]
            long_mom_signals[symbol] = sum(returns[-self._long_window:]) / self._long_window

        closes_short = view.closes(lookback=self._short_window)
        if closes_short.height < self._short_window:
            return Signal(information_available_at=stamp, weights={})

        short_mom_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes_short.columns:
                continue
            returns = [float(v) for v in (closes_short[symbol].drop_nulls().to_list()[1:] / closes_short[symbol].drop_nulls().to_list()[:-1])]
            short_mom_signals[symbol] = sum(returns[-self._short_window:]) / self._short_window

        combined_signal: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in long_mom_signals or symbol not in short_mom_signals:
                continue
            combined_signal[symbol] = (long_mom_signals[symbol] + short_mom_signals[symbol]) / 2.0

        top_symbols = sorted(combined_signal.items(), key=lambda x: x[1], reverse=True)
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [symbol for symbol, _ in top_symbols[:5]]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest