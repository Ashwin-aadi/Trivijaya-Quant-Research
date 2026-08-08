from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility is often correlated with future returns. High volatility can indicate "
        "that the market is in a trend. By following the direction of past trends and using "
        "volatility as a scaling factor, we can potentially capitalize on these trends."
    )

    def __init__(self, window: int = 20, trend_weight: float = 1.5) -> None:
        self._window = window
        self._trend_weight = trend_weight

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols_with_trends = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            trend_score = (
                (recent_closes[-1] - recent_closes[0]) / max(recent_closes)
                * self._trend_weight
            )
            if trend_score > 0:
                symbols_with_trends.append(symbol)

        top_symbols = sorted(symbols_with_trends, key=lambda s: -trend_score[s])
        top_symbols = top_symbols[:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest