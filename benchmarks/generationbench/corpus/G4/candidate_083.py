from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ITHealthcareComposite(Strategy):
    rationale = (
        "This strategy exploits the divergence in price movements between sectors with "
        "low correlation, such as IT and healthcare. By combining metrics from both sectors, "
        "we aim to capture opportunities during macroeconomic uncertainties or shifts in investor sentiment."
    )

    def __init__(self, it_window: int = 30, health_window: int = 60) -> None:
        self._it_window = it_window
        self._health_window = health_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._it_window, self._health_window))
        if closes.height < max(self._it_window, self._health_window):
            return Signal(information_available_at=stamp, weights={})

        it_signals = {}
        health_signals = {}

        for symbol in view.symbols:
            it_closes = closes.select([pl.col(symbol).alias("close")])
            health_closes = closes.select([pl.col(symbol).alias("close")])

            if "IT" in symbol:
                it_signal = (it_closes["close"].mean() - it_closes["close"].shift(self._it_window)).sum()
                it_signals[symbol] = float(it_signal)
            elif "Healthcare" in symbol:
                health_signal = (health_closes["close"].mean() - health_closes["close"].shift(self._health_window)).sum()
                health_signals[symbol] = float(health_signal)

        combined_scores = {s: 0.5 * it_signals.get(s, 0) + 0.5 * health_signals.get(s, 0) for s in view.symbols}

        top_stocks = sorted(combined_scores.items(), key=lambda x: -x[1])[:20]

        weights = {stock: score / sum(combined_scores.values()) for stock, score in top_stocks}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest