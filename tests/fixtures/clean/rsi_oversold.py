"""Buy names whose relative strength index marks them oversold."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import daily_returns, equal_weight, latest_visible


class RsiOversold(Strategy):
    """Wilder's RSI, holding names below a low threshold."""

    rationale = (
        "RSI compares the size of recent gains to recent losses. A low reading means selling has "
        "dominated for a sustained stretch, which short-horizon reversal research suggests is "
        "often followed by stabilisation."
    )

    def __init__(self, window: int = 14, threshold: float = 30.0) -> None:
        self._window = window
        self._threshold = threshold

    @staticmethod
    def _rsi(returns: list[float]) -> float:
        gains = [r for r in returns if r > 0]
        losses = [-r for r in returns if r < 0]
        average_gain = sum(gains) / len(returns) if returns else 0.0
        average_loss = sum(losses) / len(returns) if returns else 0.0
        if average_loss == 0:
            return 100.0 if average_gain > 0 else 50.0
        strength = average_gain / average_loss
        return 100.0 - 100.0 / (1.0 + strength)

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._window)
        picks = sorted(
            symbol for symbol, rets in series.items()
            if len(rets) >= self._window and self._rsi(rets[-self._window:]) < self._threshold
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
