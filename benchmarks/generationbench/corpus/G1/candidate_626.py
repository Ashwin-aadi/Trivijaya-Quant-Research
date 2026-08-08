from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends in the presence of volatility. When a stock's price "
        "moves strongly in one direction relative to its recent volatility, it signals that "
        "the trend is likely to continue."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = {symbol: float(v) for symbol, v in view.latest_close().items()}
        trend_signals: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            adj_closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"].to_list()]
            if len(adj_closes) < self._window:
                continue

            trend = (adj_closes[-1] - adj_closes[0]) / sum(abs(a - b) for a, b in zip(adj_closes[:-1], adj_closes[1:]))
            volatility = pl.DataFrame({"close": adj_closes}).with_column(
                (pl.col("close").rolling_mean(window=self._window) -
                 pl.col("close")).abs().mean()
            ).select(pl.col("close").to_list())[0]
            
            if trend > 2 * volatility:
                trend_signals[symbol] = 1.0
            elif trend < -2 * volatility:
                trend_signals[symbol] = -1.0

        weighted_trend_signal: float = sum(trend_signals.values())
        if not trend_signals:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = weighted_trend_signal / len(trend_signals)
        return Signal(
            information_available_at=stamp,
            weights={symbol: max(-1.0, min(1.0, weight_per_symbol)) for symbol in trend_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest