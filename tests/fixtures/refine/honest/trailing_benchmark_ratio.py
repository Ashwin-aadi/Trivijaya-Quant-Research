"""Hold names whose trailing growth beats the universe's trailing average by a stated ratio."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, window_return


class TrailingBenchmarkRatio(Strategy):
    """Divides each name's trailing growth factor by the equal-weighted universe's own."""

    rationale = (
        "Measuring a stock against its universe rather than against zero removes the move common "
        "to all of them, leaving the part specific to the name. Expressing it as a ratio of "
        "growth factors rather than a difference of returns keeps the comparison meaningful when "
        "the benchmark itself moved a long way. The hurdle above one is a deliberate dead band so "
        "the portfolio does not churn over names that merely matched the average."
    )

    def __init__(self, window: int = 126, ratio: float = 1.05) -> None:
        if ratio <= 0.0:
            raise ValueError("the ratio must be positive")
        self._window = window
        self._ratio = ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        returns = window_return(view, self._window)
        if len(returns) < 2:
            return Signal(information_available_at=stamp, weights={})

        # The benchmark is the equal-weighted average of the same trailing window over the same
        # point-in-time universe, so both sides of the ratio end on the last visible session and
        # both are drawn from the constituents as they stood on that date.
        benchmark_average = sum(returns.values()) / len(returns)
        if 1.0 + benchmark_average <= 0.0:
            return Signal(information_available_at=stamp, weights={})

        picks = sorted(
            symbol
            for symbol, value in returns.items()
            if (1.0 + value) / (1.0 + benchmark_average) >= self._ratio
        )
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
