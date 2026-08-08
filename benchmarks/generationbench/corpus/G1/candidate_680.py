from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment. A significant "
        "volume increase on a price breakout often signals a sustained trend in the direction of "
        "the move."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes().to_numpy()
        symbols = [s for s in view.symbols if len(latest_closes[:, latest_closes[0] == s]) >= self._window]

        signals: dict[str, float] = {}
        for symbol in symbols:
            adj_closes = history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()
            prices = [float(v) for v in adj_closes]
            volumes = (
                view.history(lookback=self._window)
                .filter(pl.col("symbol") == symbol)
                .select("volume")
                .to_numpy()
            )
            vol_changes = [float(v) for v in volumes]

            if len(prices) < self._window or len(vol_changes) < self._window:
                continue

            breakout_index = prices.index(max(prices))
            pre_breakout_price = prices[breakout_index - 1]
            post_breakout_volume = vol_changes[breakout_index + 1]
            pre_breakout_volume = vol_changes[breakout_index]

            if max(prices) > pre_breakout_price and post_breakout_volume > pre_breakout_volume:
                signals[symbol] = 1.0 / len(symbols)

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest