"""Hold names outperforming the universe average over the lookback."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, window_return


class RelativeStrengthVsUniverse(Strategy):
    """Compares each name against the equal-weighted universe return."""

    rationale = (
        "Measuring a stock against its own universe rather than against zero removes the market "
        "move common to all of them, so what remains is the part specific to that name. Only "
        "names beating the contemporaneous average are held."
    )

    def __init__(self, lookback: int = 63) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._lookback)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        # The benchmark is this date's cross-sectional mean, computed from visible data only.
        average = sum(returns.values()) / len(returns)
        picks = sorted(symbol for symbol, value in returns.items() if value > average)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
