from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies exploit the fact that assets with higher "
        "volatility tend to exhibit larger price movements over a given period. By scaling our "
        "trends based on historical volatility, we can capture more significant movements while "
        "attempting to avoid mean-reverting periods."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        returns = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
        volatility = ((pl.Series(returns).abs() * (252 / self._window)).mean()).get()
        
        trend_strengths: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            series = pl.DataFrame(history[[symbol]]).select(
                "session_date", (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            mean_return = series.select(pl.col("return").mean()).item()
            trend_strengths[symbol] = abs(mean_return) * volatility

        sorted_symbols = [k for k, v in sorted(trend_strengths.items(), key=lambda item: item[1], reverse=True)]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest