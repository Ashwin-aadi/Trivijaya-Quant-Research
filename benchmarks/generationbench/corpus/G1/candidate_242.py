from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Trends in asset prices are more likely to continue than reverse. By scaling trades by "
        "volatility, we can potentially benefit from these trends while limiting risk."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._volatility_window - 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if all(s in history.columns)]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        volatilities: dict[str, float] = {}

        for symbol in symbols:
            closes = history.select(pl.col(symbol)).to_numpy().flatten()
            trend = (closes[-1] - closes[0]) / self._window
            volatility = pl.DataFrame(closes).select(
                (pl.col(0) - pl.col(0).mean()).abs().sum() / self._volatility_window
            ).item()
            trends[symbol] = trend
            volatilities[symbol] = volatility

        ranked_trends = sorted(trends.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [s for s, _ in ranked_trends[: self._volatility_window]]

        weights = {}
        for symbol in top_symbols:
            weight = 1.0 / len(top_symbols) * (abs(trends[symbol]) / volatilities[symbol])
            if symbol not in trends or symbol not in volatilities:
                continue
            weights[symbol] = max(min(weight, 0.5), -0.5)

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if abs(w) > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest