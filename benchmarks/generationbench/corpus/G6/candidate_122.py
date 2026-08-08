from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capture upward price movements during trends while "
        "reducing exposure during consolidations or reversals based on volatility. "
        "It uses 50-day and 200-day simple moving averages (SMA) for trend identification, "
        "combined with realized volatility to gauge market conditions."
    )

    def __init__(self, window_50: int = 50, window_200: int = 200, window_volatility: int = 30, max_stocks: int = 10) -> None:
        self._window_50 = window_50
        self._window_200 = window_200
        self._window_volatility = window_volatility
        self._max_stocks = max_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_50 + self._window_200 + self._window_volatility)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_50 = (history["adj_close"] / history["adj_close"].shift(self._window_50).fill_null(1) - 1).mean().item()
        sma_200 = (history["adj_close"] / history["adj_close"].shift(self._window_200).fill_null(1) - 1).mean().item()

        if abs(sma_50 - sma_200) < 0.003:
            return Signal(information_available_at=stamp, weights={})

        volatility = (history["adj_close"].pct_change().rolling_std(self._window_volatility)).mean().item()
        signals: list[str] = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            sma_50_symbol = (history[symbol]["adj_close"] / history[symbol]["adj_close"].shift(self._window_50).fill_null(1) - 1).mean().item()
            sma_200_symbol = (history[symbol]["adj_close"] / history[symbol]["adj_close"].shift(self._window_200).fill_null(1) - 1).mean().item()

            if abs(sma_50_symbol - sma_200_symbol) > 0.003 and volatility < 0.03:
                signals.append(symbol)

        picks = sorted(signals, key=lambda x: (sma_50_history[x] / sma_200_history[x], volatility[x]), reverse=True)[:self._max_stocks]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest