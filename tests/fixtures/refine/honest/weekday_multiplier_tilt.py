"""Hold the universe at an exposure that depends on the weekday of the last visible session."""

from __future__ import annotations

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible

# One multiplier per weekday under date.weekday() numbering, Monday first. Exposure is trimmed
# into the weekend gap and set to zero on Saturday and Sunday, which never occur as sessions but
# keep the tuple aligned with the numbering rather than requiring an offset at the call site.
DEFAULT_WEEKDAY_TILT: tuple[float, ...] = (1.0, 1.0, 1.0, 0.75, 0.5, 0.0, 0.0)


class WeekdayMultiplierTilt(Strategy):
    """Scales an equal-weighted universe holding by a fixed per-weekday multiplier."""

    rationale = (
        "Two non-trading days let news accumulate with no chance to react, so carrying a smaller "
        "book into the weekend reduces gap risk. That is a reduction in exposure, not a source of "
        "return: over a long sample this should give up part of the market's drift in exchange "
        "for a slightly smaller worst week. Day-of-week return effects themselves are among the "
        "most heavily data-mined regularities in equities and no weight is placed on them here."
    )

    def __init__(self, weekday_tilt: tuple[float, ...] = DEFAULT_WEEKDAY_TILT) -> None:
        if len(weekday_tilt) != 7:
            raise ValueError("one multiplier is needed for each weekday")
        if any(not 0.0 <= multiplier <= 1.0 for multiplier in weekday_tilt):
            raise ValueError("multipliers must lie in [0, 1] so gross exposure stays within one")
        # Seven constants. This is a setting, not a panel: it holds no prices, no dates, and
        # nothing estimated from the sample, so nothing about the future can hide in it.
        self._tilt = weekday_tilt

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        if not view.symbols or view.history(lookback=1).is_empty():
            return Signal(information_available_at=stamp, weights={})

        multiplier = self._tilt[stamp.weekday()]
        if multiplier <= 0.0:
            return Signal(information_available_at=stamp, weights={})
        base = equal_weight(sorted(view.symbols))
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight * multiplier for symbol, weight in base.items()},
        )
