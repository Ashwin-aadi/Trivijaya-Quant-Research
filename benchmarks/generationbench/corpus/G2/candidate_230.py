from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following is based on the idea that during trending markets, "
        "high-volatility stocks are more likely to continue their trend. This strategy aims to "
        "capitalize on such trends by allocating capital to high-volatility stocks in upward "
        "trending periods."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._volatility_window)

        if history.is_empty() or history.height < self._window + self._volatility_window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._volatility_window)
        symbols = view.symbols

        volatility = {
            symbol: float(closes[symbol].std()) for symbol in symbols
        }
        trend = _calculate_trend(view.history())

        ranked_symbols = sorted(
            zip(symbols, [volatility[s] * trend.get(symbol, 0.0) for symbol in symbols]),
            key=lambda x: -x[1],
        )[:5]

        weights = {s: w / sum([w for _, w in ranked_symbols]) for s, w in ranked_symbols}
        if not weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight
                for symbol, _ in ranked_symbols
                for symbol, weight in weights.items()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_trend(df: pl.DataFrame) -> dict[str, float]:
    df = df.sort("session_date")
    trend = {}
    for symbol in view.symbols:
        if symbol not in df.columns:
            continue
        series = [float(v) for v in df[symbol].to_list()]
        if len(series) < 20:
            continue
        slope, intercept, _, _, _ = pl.fit_regression(
            pl.Series("t", range(1, len(series) + 1)), pl.Series(symbol, series)
        )
        trend[symbol] = slope.to_numpy()[0]
    return trend