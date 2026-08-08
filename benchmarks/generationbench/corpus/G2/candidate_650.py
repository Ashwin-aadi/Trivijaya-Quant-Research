from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High historical volatility suggests that a security is currently experiencing price "
        "fluctuations driven by significant market events or news. By following trends in "
        "highly volatile stocks, we can capture potentially profitable movements before they "
        "return to more normal conditions."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volatility = _rolling_volatility(closes)
            trend_strength = (closes[-1] - closes[0]) / max(volatility, 1e-6)

            if trend_strength > self._threshold:
                trends[symbol] = trend_strength

        if not trends:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(trends)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in trends.keys()},
        )


def _rolling_volatility(closes: list[float]) -> float:
    returns = [c2 - c1 for c1, c2 in zip(closes[:-1], closes[1:])]
    return (sum([r**2 for r in returns]) / len(returns)) ** 0.5


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest