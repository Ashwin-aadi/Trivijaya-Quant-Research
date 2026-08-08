from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeTrendVolume(Strategy):
    rationale = (
        "Combining trend strength with volume can help identify stocks that are not only moving "
        "in a favorable direction but also doing so with significant liquidity. This dual criterion"
        "can reduce the risk of chasing false trends."
    )

    def __init__(self, trend_window: int = 20, volume_threshold: float = 1_000_000) -> None:
        self._trend_window = trend_window
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window)
        if closes.height < self._trend_window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the trend strength
        trends = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._trend_window:
                continue
            trend_strength = (values[-1] - values[0]) / values[0]
            trends[symbol] = trend_strength

        # Filter by volume and select top n symbols based on combined criteria
        picks: list[str] = []
        for symbol in view.symbols:
            history = view.history(lookback=self._trend_window)
            if symbol not in history.columns or history.height < self._trend_window:
                continue
            volume_data = [float(v) for v in history[symbol].select("volume").drop_nulls().to_list()]
            if len(volume_data) < self._trend_window:
                continue
            recent_volume_mean = pl.Series(volume_data).mean()
            if recent_volume_mean > self._volume_threshold and symbol in trends:
                picks.append(symbol)

        # Limit the number of selected symbols
        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest