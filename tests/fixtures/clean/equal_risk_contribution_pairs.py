"""Split capital between a calm and a volatile half so each contributes similar risk."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, latest_visible, stdev


def _halves(volatilities: dict[str, float]) -> tuple[list[str], list[str]]:
    """Split the universe at its median volatility into a calm basket and a volatile one."""
    # Symbol breaks ties so the split is reproducible when volatilities coincide.
    ordered = sorted(volatilities.items(), key=lambda kv: (kv[1], kv[0]))
    cut = len(ordered) // 2
    return [s for s, _ in ordered[:cut]], [s for s, _ in ordered[cut:]]


class EqualRiskContributionPairs(Strategy):
    """Two baskets, sized inversely to their average volatility."""

    rationale = (
        "Holding a calm basket and a volatile basket in equal capital gives the volatile one "
        "almost all of the portfolio's risk, so the result is really a bet on high-volatility "
        "stocks. Sizing the two inversely to their average volatility equalises what each "
        "contributes. Correlation between the baskets is ignored, so this is the naive form of "
        "risk parity rather than the solved one, and the equalisation is only approximate."
    )

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        volatilities = {
            symbol: stdev(returns)
            for symbol, returns in series.items()
            if len(returns) >= 2 and stdev(returns) > 0
        }
        if len(volatilities) < 2:
            return Signal(information_available_at=stamp, weights={})

        calm, volatile = _halves(volatilities)
        if not calm or not volatile:
            return Signal(information_available_at=stamp, weights={})
        calm_level = sum(volatilities[s] for s in calm) / len(calm)
        volatile_level = sum(volatilities[s] for s in volatile) / len(volatile)
        total_level = calm_level + volatile_level
        if total_level <= 0:
            return Signal(information_available_at=stamp, weights={})

        # Capital inversely proportional to each basket's volatility reduces to this share, and
        # the two shares sum to one, so gross exposure is exactly one.
        calm_share = volatile_level / total_level
        weights = {s: calm_share / len(calm) for s in calm}
        weights.update({s: (1.0 - calm_share) / len(volatile) for s in volatile})
        return Signal(information_available_at=stamp, weights=weights)
