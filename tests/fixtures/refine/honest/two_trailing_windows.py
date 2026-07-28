"""Hold names whose recent volatility has fallen well below their own longer-run volatility."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import daily_returns, equal_weight, latest_visible, stdev


class TwoTrailingWindows(Strategy):
    """Compares a short trailing volatility window against a long one ending on the same day."""

    rationale = (
        "Volatility clusters and mean-reverts, so a name unusually quiet against its own norm is "
        "more likely than not to see that quiet end. Holding such names long additionally assumes "
        "the resolution is upward, which the compression itself gives no reason to expect — that "
        "assumption is the weakest part of the idea and it is stated rather than hidden. The "
        "ratio threshold is a round number and has not been searched over."
    )

    def __init__(self, short_window: int = 10, long_window: int = 63, ratio: float = 0.8) -> None:
        if short_window >= long_window:
            raise ValueError("the short window must be shorter than the long window")
        if not 0.0 < ratio < 1.0:
            raise ValueError("the ratio must lie in (0, 1) for this to mean compression")
        self._short = short_window
        self._long = long_window
        self._ratio = ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        series = daily_returns(view, self._long)

        picks: list[str] = []
        for symbol, returns in sorted(series.items()):
            if len(returns) < self._long:
                continue
            # Both windows end on the last visible session and the short one is a suffix of the
            # long one, so the comparison is between two pasts of different length. Neither
            # window reaches forward of the decision moment.
            recent_window = returns[-self._short:]
            baseline_window = returns[-self._long:]
            recent_volatility = stdev(recent_window)
            baseline_volatility = stdev(baseline_window)
            if baseline_volatility <= 0:
                continue
            if recent_volatility < baseline_volatility * self._ratio:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
