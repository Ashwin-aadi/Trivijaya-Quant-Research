from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolatility(Strategy):
    rationale = (
        "This strategy combines two signals: a momentum signal based on 50-day returns and "
        "a volatility signal based on 20-day standard deviation. The idea is that stocks with strong"
        "momentum but low recent volatility are more likely to continue their trend."
    )

    def __init__(self, momentum_window: int = 50, vol_window: int = 20) -> None:
        self._momentum_window = momentum_window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._momentum_window + self._vol_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        symbols = [s for s in view.symbols if s in closes.columns]

        momentum_scores: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for symbol in symbols:
            close_series = closes[symbol].drop_nulls()
            recent_closes = close_series.tail(self._momentum_window)
            recent_open = close_series.shift(-1).tail(self._momentum_window)

            if len(recent_closes) < self._momentum_window or len(recent_open) < self._momentum_window:
                continue

            returns = (recent_closes - recent_open) / recent_open
            momentum_score = returns.mean().item()
            momentum_scores[symbol] = momentum_score

            vol_series = close_series.tail(self._vol_window)
            if vol_series.height < self._vol_window:
                continue
            volatility = vol_series.std().item()
            volatilities[symbol] = volatility

        filtered_symbols = [
            s for s in symbols if s in momentum_scores and s in volatilities
        ]

        final_signals: dict[str, float] = {}
        for symbol in filtered_symbols:
            momentum_score = momentum_scores[symbol]
            volatility = volatilities[symbol]

            signal_strength = (momentum_score / 10.0) * (2 - volatility)
            if signal_strength > 1.0:
                final_signals[symbol] = signal_strength

        top_n = min(len(final_signals), 5)
        sorted_final_signals = sorted(final_signals.items(), key=lambda x: x[1], reverse=True)

        picks = [symbol for symbol, _ in sorted_final_signals[:top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest