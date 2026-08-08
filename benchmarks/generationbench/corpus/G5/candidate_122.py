from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to capture trends by scaling positions "
        "based on historical volatility. High volatility periods suggest increased risk, "
        "thus reducing exposure; low volatility periods indicate reduced risk, allowing for "
        "larger positions."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.0) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 - 1:
            return Signal(information_available_at=stamp, weights={})

        volatility = _compute_volatility(closes)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in volatility.columns:
                continue
            latest_vol = float(volatility[symbol][-1])
            if latest_vol < 0.2 * self._scale_factor:
                picks.append(symbol)
            else:
                weight = min(1.0, (latest_vol / self._scale_factor) ** -self._window)
                picks.append((symbol, weight))

        weights = {s: w for s, w in picks if w > 0}
        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _compute_volatility(closes: pl.DataFrame) -> pl.DataFrame:
    returns = (closes / closes.shift(1).fill_null(1.0) - 1.0).drop_nulls().sort("session_date").tail(self._window).select(pl.all().mean().alias("volatility"))
    return returns