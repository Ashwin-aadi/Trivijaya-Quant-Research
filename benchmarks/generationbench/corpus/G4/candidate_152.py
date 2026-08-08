from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy exploits volume-confirmed directional moves by identifying stocks "
        "with significant increases or decreases in trading volume accompanied by corresponding"
        " price movements. High trading volumes often indicate strong investor sentiment, "
        "potentially validating existing trends and leading to further momentum."
    )

    def __init__(self, window: int = 20, min_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._min_volume_ratio = min_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2:  # To ensure we have enough data
            return Signal(information_available_at=stamp, weights={})

        buys = []
        sells = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            volume_changes = (history[f"{symbol}_volume"] / history[f"{symbol}_volume"].shift(1) - 1.0).alias("volume_change")
            close_changes = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("close_change")

            volume_changes_df = (
                history.with_columns(volume_changes)
                       .sort("session_date", descending=True)
                       .tail(self._window + 1)
            )

            close_changes_df = (
                history.with_columns(close_changes)
                       .sort("session_date", descending=True)
                       .tail(self._window + 1)
            )

            volume_changes_list = [float(v) for v in volume_changes_df["volume_change"].to_list()]
            close_changes_list = [float(v) for v in close_changes_df["close_change"].to_list()]

            if len(volume_changes_list) < self._window or len(close_changes_list) < self._window:
                continue

            recent_volume_change = max(volume_changes_list)
            recent_close_change = max(close_changes_list)

            if (recent_volume_change > self._min_volume_ratio and
                    recent_close_change >= 0.01):
                buys.append(symbol)

            if (recent_volume_change > self._min_volume_ratio and
                    recent_close_change <= -0.01):
                sells.append(symbol)

        buys = sorted(buys, key=lambda x: max(volume_changes_list[:3]), reverse=True)[:20]
        sells = sorted(sells, key=lambda x: min(close_changes_list[:3]), reverse=False)[:20]

        if not buys and not sells:
            return Signal(information_available_at=stamp, weights={})

        buy_weights = {symbol: 0.05 / len(buys) for symbol in buys}
        sell_weights = {symbol: -0.05 / len(sells) for symbol in sells}

        all_weights = {}
        for symbol in buys:
            if symbol not in all_weights:
                all_weights[symbol] = buy_weights[symbol]
            else:
                all_weights[symbol] += buy_weights[symbol]

        for symbol in sells:
            if symbol not in all_weights:
                all_weights[symbol] = sell_weights[symbol]
            else:
                all_weights[symbol] -= sell_weights[symbol]

        return Signal(
            information_available_at=stamp, weights=all_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest