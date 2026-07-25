"""Hold names whose recent volatility has risen above their own longer-run level."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, equal_weight, latest_visible, stdev


class VolatilityBreakout(Strategy):
    """Selects on a rise in a name's own realised volatility, not on price direction."""

    rationale = (
        "Volatility clusters, so a stretch of unusually large moves tends to be followed by more "
        "of the same. Comparing a name's recent volatility against its own longer-run level "
        "detects that a quiet regime has ended without reference to the size of the stock. The "
        "rule holds those names long, which assumes the expansion resolves upward — the "
        "volatility measure itself says nothing about direction."
    )

    def __init__(
        self, short_window: int = 21, long_window: int = 126, ratio: float = 1.25
    ) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        self._short = short_window
        self._long = long_window
        self._ratio = ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._long)

        picks: list[str] = []
        for symbol, returns in series.items():
            if len(returns) < self._long:
                continue
            recent = stdev(returns[-self._short:])
            baseline = stdev(returns)
            if baseline > 0 and recent > self._ratio * baseline:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(sorted(picks)))
