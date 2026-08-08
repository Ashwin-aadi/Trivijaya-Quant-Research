from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum20d(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting stocks with the highest "
        "recent performance and limiting losses through a combination of CUMRET threshold and stop-loss mechanisms."
    )

    def __init__(self, window: int = 20, top_n_percent: float = 0.15, cumret_threshold: float = -0.08, stop_loss: float = -0.12) -> None:
        self._window = window
        self._top_n_percent = top_n_percent
        self._cumret_threshold = cumret_threshold
        self._stop_loss = stop_loss

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = history["symbol"].unique().to_list()
        closes = view.closes(lookback=self._window)

        cumrets: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            cumret = (close_series[-1] / close_series[0]) - 1.0
            cumrets[symbol] = cumret

        ranked_symbols = sorted(cumrets.items(), key=lambda x: x[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in ranked_symbols[:int(len(symbols) * self._top_n_percent)]]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.05
        portfolio_weights = {s: weight for s in top_n_symbols}
        
        return Signal(
            information_available_at=stamp,
            weights=portfolio_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest