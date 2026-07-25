"""Hold the whole universe, weighted inversely to each name's volatility."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, latest_visible, stdev


class InverseVolatilityWeighted(Strategy):
    """Risk-parity style sizing, with no selection component."""

    rationale = (
        "Equal capital weights give volatile names a disproportionate share of portfolio risk. "
        "Sizing inversely to volatility equalises each position's risk contribution instead, "
        "which is the standard naive risk-parity construction."
    )

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        inverse = {
            symbol: 1.0 / stdev(rets)
            for symbol, rets in series.items()
            if len(rets) >= 2 and stdev(rets) > 0
        }
        total = sum(inverse.values())
        if total <= 0:
            return Signal(information_available_at=stamp, weights={})
        # Normalised so gross exposure is exactly one.
        weights = {symbol: value / total for symbol, value in inverse.items()}
        return Signal(information_available_at=stamp, weights=weights)
