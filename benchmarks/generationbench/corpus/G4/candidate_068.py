from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy identifies strong directional moves in the market that are "
        "volume-confirmed. By combining price and volume signals, we aim to capture "
        "sustained trends with reduced risk."
    )

    def __init__(self, window_price: int = 5, threshold_volume: float = 0.1) -> None:
        self._window_price = window_price
        self._threshold_volume = threshold_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_price + 1)
        if history.height < self._window_price + 1:
            return Signal(information_available_at=stamp, weights={})

        price_changes = {}
        volume_changes = {}

        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue

            prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "adj_close"
            ].drop_nulls().to_list()]
            volumes = [float(v) for v in history.filter(pl.col("symbol") == symbol)[
                "volume"
            ].drop_nulls().to_list()]

            if len(prices) < self._window_price + 1:
                continue

            # Calculate price and volume changes
            recent_close = prices[-1]
            sma_upside = sum(prices[max(0, -self._window_price - 1): -1]) / self._window_price
            sma_downside = min(prices[: -self._window_price])
            is_upside_move = recent_close > sma_upside and prices[-2] < sma_upside
            is_downside_move = recent_close < sma_downside and prices[-2] > sma_downside

            volume_change = (volumes[-1] / volumes[-2] - 1.0) if len(volumes) > 1 else 0.0
            price_changes[symbol] = is_upside_move or is_downside_move
            volume_changes[symbol] = volume_change >= self._threshold_volume

        # Filter symbols with both a directional move and significant volume increase
        candidates = [s for s in price_changes if price_changes[s] and volume_changes[s]]

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_pydatetime().date()
    assert isinstance(newest, date)
    return newest