from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeVolatilityMomentum(Strategy):
    rationale = (
        "This strategy combines high historical volatility with recent momentum to identify "
        "potentially strong and volatile stocks in the NIFTY 100 index. High volatility often "
        "indicates a stock with significant price movement, which can be profitable if the "
        "momentum is positive."
    )

    def __init__(self, window_volatility: int = 20, threshold_volatility: float = 0.5) -> None:
        self._window_volatility = window_volatility
        self._threshold_volatility = threshold_volatility

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_volatility)
        if closes.height < self._window_volatility:
            return Signal(information_available_at=stamp, weights={})

        volatility_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_volatility:
                continue

            daily_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            volatility_score = pl.Series(daily_returns).abs().mean()
            if volatility_score >= self._threshold_volatility:
                volatility_scores[symbol] = volatility_score

        if not volatility_scores:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [k for k, v in sorted(volatility_scores.items(), key=lambda item: -item[1])]
        top_n = min(len(sorted_symbols), 5)
        picks = sorted_symbols[:top_n]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest