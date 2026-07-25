"""Hold the least volatile names in the universe."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, equal_weight, latest_visible, stdev, top_n


class LowVolatility(Strategy):
    """Selects on realised volatility, lowest first."""

    rationale = (
        "The low-volatility anomaly is the observation that calmer stocks have historically "
        "delivered risk-adjusted returns at least as good as volatile ones, contrary to what a "
        "simple risk-return tradeoff would predict."
    )

    def __init__(self, lookback: int = 63, holdings: int = 10) -> None:
        self._lookback = lookback
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._lookback)
        vols = {sym: stdev(rets) for sym, rets in series.items() if len(rets) >= 2}
        vols = {sym: v for sym, v in vols.items() if v > 0}
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(vols, self._holdings, largest=False)),
        )
