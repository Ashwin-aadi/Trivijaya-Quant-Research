"""Hold constituents whose trailing return cleared a threshold set in a frozen settings object."""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.strategy import MarketView, Signal, Strategy

from ...clean._common import equal_weight, latest_visible, window_return


@dataclass(frozen=True)
class TrendSettings:
    """Every knob this strategy has. Numbers only: no prices, no dates, no frames."""

    window: int = 126
    min_return: float = 0.0
    min_names: int = 3


class FrozenSettingsTrend(Strategy):
    """Absolute-momentum filter whose parameters arrive as one immutable settings object."""

    rationale = (
        "Absolute momentum holds only what has already risen over the lookback, which keeps the "
        "portfolio out of names in sustained decline and moves it to cash when most of the "
        "universe is falling. The minimum-names rule prevents the portfolio from concentrating "
        "into two or three survivors during a broad drawdown, at the cost of missing the early "
        "part of a recovery when only a handful of names have turned."
    )

    def __init__(self, settings: TrendSettings | None = None) -> None:
        # The constructor takes settings, never market data. Grouping them in a frozen dataclass
        # means a caller cannot mutate the configuration between decision dates, which would make
        # a run irreproducible without leaving any trace in the manifest.
        self._settings = settings if settings is not None else TrendSettings()

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        settings = self._settings
        returns = window_return(view, settings.window)
        picks = sorted(
            symbol for symbol, value in returns.items() if value > settings.min_return
        )
        if len(picks) < settings.min_names:
            return Signal(information_available_at=stamp, weights={})
        return Signal(information_available_at=stamp, weights=equal_weight(picks))
