"""Hold names whose MACD line sits above its own signal line."""

from __future__ import annotations

from collections.abc import Sequence

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible


def _ema(values: Sequence[float], span: int) -> list[float]:
    """Exponential moving average over ``values``, seeded on the first observation.

    Seeding on the first point rather than on a simple average of the first ``span`` points is the
    convention most charting packages use; the difference washes out once enough sessions have
    passed, which is why the caller demands a long warm-up window.
    """
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    level = values[0]
    out: list[float] = []
    for value in values:
        level = alpha * value + (1.0 - alpha) * level
        out.append(level)
    return out


class MacdSignalCross(Strategy):
    """Trend following on the standard 12/26/9 MACD construction."""

    rationale = (
        "The MACD line is the gap between a fast and a slow exponential average, so it is "
        "positive when recent prices are pulling away from the established level. Comparing it "
        "against its own smoothed version asks whether that gap is still widening rather than "
        "merely positive, which is a slower entry than a raw moving-average cross."
    )

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        if fast >= slow:
            raise ValueError("the fast span must be shorter than the slow span")
        self._fast = fast
        self._slow = slow
        self._signal = signal
        # Three slow spans of warm-up: the weight an exponential average places on its seed decays
        # geometrically, and by then it is small enough not to drive the sign of the cross.
        self._lookback = 3 * slow

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue
            fast_line = _ema(values, self._fast)
            slow_line = _ema(values, self._slow)
            macd = [f - s for f, s in zip(fast_line, slow_line, strict=True)]
            if macd[-1] > _ema(macd, self._signal)[-1]:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
