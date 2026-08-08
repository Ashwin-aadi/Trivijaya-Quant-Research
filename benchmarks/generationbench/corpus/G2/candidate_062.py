from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies aim to capture gains from trending "
        "markets by adjusting the position size based on recent volatility. High volatility "
        "indicates a more uncertain price movement, which may be followed by a reversal or continuation. "
        "By scaling positions according to historical volatility, we can potentially capitalize on "
        "these trends without overexposing during volatile periods."
    )

    def __init__(self, window: int = 20, factor: float = 1.0) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.symbol.unique().to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in symbols:
            symbol_history = history.select(["session_date", pl.col(symbol).alias("adj_close")])
            returns = (symbol_history["adj_close"] / symbol_history["adj_close"].shift(1) - 1.0).drop_nulls()
            if returns.height < self._window + 1:
                continue

            volatility = returns.std() * self._factor
            latest_return = float(symbol_history.select("adj_close").tail(2)[-1] / symbol_history.tail(3)[0]["adj_close"] - 1)
            weight = latest_return / (volatility + 1e-8)  # Avoid division by zero

            signals[symbol] = max(min(weight, 1.0), 0.0)

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signals.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest