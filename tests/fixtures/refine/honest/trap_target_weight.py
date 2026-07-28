"""Size each position inversely to its trailing volatility, capped per name.

Naming note: the per-name allocation is called ``target_weight`` because a target weight is what
a portfolio-construction step produces — the holding it is aiming at before the engine trades
towards it. ``target`` here means a desired position, not a prediction target, and no forward
return is computed anywhere in this file.
"""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import daily_returns, latest_visible, stdev


class TargetWeightSizing(Strategy):
    """Inverse-volatility weights over the visible window, capped, with the remainder in cash."""

    rationale = (
        "Equal capital across names of unequal volatility hands the noisiest few most of the "
        "portfolio's risk. Weighting each name by the reciprocal of its trailing volatility "
        "equalises the contributions approximately — approximately, because correlation between "
        "the names is ignored, so this is the naive form of risk parity rather than the solved "
        "one. A per-name cap stops a single very quiet stock from absorbing most of the book, and "
        "whatever the cap trims is held as cash instead of being redistributed."
    )

    def __init__(self, lookback: int = 63, cap: float = 0.2) -> None:
        if not 0.0 < cap <= 1.0:
            raise ValueError("the per-name cap must lie in (0, 1]")
        self._lookback = lookback
        self._cap = cap

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        inverse: dict[str, float] = {}
        for symbol, returns in series.items():
            volatility = stdev(returns)
            if volatility > 0:
                inverse[symbol] = 1.0 / volatility
        total = sum(inverse.values())
        if total <= 0:
            return Signal(information_available_at=stamp, weights={})

        weights: dict[str, float] = {}
        for symbol, score in sorted(inverse.items()):
            # Each name's share of the inverse-volatility total, then clipped. Clipping only ever
            # reduces a share, so the weights sum to at most one and gross exposure stays within
            # the engine's limit without a renormalisation step.
            target_weight = min(score / total, self._cap)
            weights[symbol] = target_weight
        return Signal(information_available_at=stamp, weights=weights)
