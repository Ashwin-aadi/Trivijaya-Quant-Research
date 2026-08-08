from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency for asset prices to revert to their "
        "historical average. In a strong mean-reverting market, large deviations from the mean "
        "tend to be corrected in short periods, providing opportunities for profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_close = sum(values) / self._window
            recent_close = values[-1]
            z_score = (recent_close - mean_close) / max(mean_close, 1e-6)
            
            if abs(z_score) >= self._threshold:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        normalized_weights = {symbol: weight / total_weight for symbol, weight in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in normalized_weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest