"""Hold the most volatile names — the deliberate inverse of the low-volatility fixture."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, equal_weight, latest_visible, stdev, top_n


class HighVolatility(Strategy):
    """Selects on realised volatility, highest first."""

    rationale = (
        "Included as the mirror image of the low-volatility rule. If the low-volatility fixture "
        "shows an effect, this one should show its opposite; if both look similar, the apparent "
        "effect is more likely an artifact of the test setup than a property of the market."
    )

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings)),
        )
