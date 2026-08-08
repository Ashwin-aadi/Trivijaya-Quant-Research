from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting for volatility. "
        "During periods of low volatility, the strategy increases exposure, benefiting from persistent "
        "trends. Conversely, during high volatility, it reduces exposure to avoid drawdowns."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window - 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        vol = _rolling_volatility(closes, window=self._vol_window)
        trend_score = _rolling_trend_strength(closes, window=self._window)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes[-self._window:]:
                continue
            score = vol[-1] * trend_score[-1]
            signals[symbol] = score

        # Normalize scores to sum up to 1.0 (or close)
        total_score = sum(signals.values())
        weights = {s: signals[s] / total_score for s in signals}
        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v > 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rolling_volatility(closes: list[float], window: int) -> float:
    returns = [c / closes[i - 1] - 1 for i, c in enumerate(closes) if i > 0]
    volatility = (pl.Series(returns).rolling_std(window=window)).mean().to_list()[-1]
    return volatility


def _rolling_trend_strength(closes: list[float], window: int) -> float:
    rolling_max = pl.Series(closes).rolling_max(window).to_list()
    trend_strength = max((c / m - 1 for c, m in zip(closes, rolling_max)), default=0)
    return trend_strength