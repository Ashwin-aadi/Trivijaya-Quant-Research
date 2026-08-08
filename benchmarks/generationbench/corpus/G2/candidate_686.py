from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to identify assets that are trending "
        "in a strong direction while controlling exposure based on volatility. High volatility "
        "periods suggest increased risk and reduced exposure is appropriate. This strategy aims "
        "to capture trends by scaling investment into highly volatile but trending assets."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_prices = [float(v) for v in history[symbol]["high"].to_list()]
            low_prices = [float(v) for v in history[symbol]["low"].to_list()]

            # Calculate daily returns and absolute changes
            returns = [(high - low) / low for high, low in zip(high_prices[1:], low_prices[:-1])]
            abs_changes = [abs(returns[i]) for i in range(len(returns))]

            # Compute average volatility over the window
            avg_volatility = sum(abs_changes) / len(abs_changes)
            symbol_volatility[symbol] = avg_volatility

        # Identify trending symbols with high returns and low volatility
        trending_symbols = [
            (symbol, returns[-1]) for symbol, returns in history.to_dict(False).items()
            if returns[-1] > 0.05 * symbol_volatility.get(symbol, 0)
        ]

        if not trending_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Scale the weights based on volatility
        total_weight = 0
        for symbol, weight in [(symbol, 1.0 / len(trending_symbols)) for _, (symbol, _) in trending_symbols]:
            total_weight += weight

        return Signal(
            information_available_at=stamp,
            weights={s: w / total_weight for s, _ in trending_symbols for w in [weight]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest