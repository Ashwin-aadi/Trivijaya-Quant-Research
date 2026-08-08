from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedBreakout(Strategy):
    rationale = (
        "This strategy identifies strong price movements accompanied by significant trading volumes "
        "to capitalize on trend-following behavior. Breakouts from consolidation patterns are used to enter positions, "
        "ensuring that a breakout is confirmed by high volume for at least three consecutive sessions."
    )

    def __init__(self, lookback_period: int = 5, max_positions: int = 20, max_weight_per_position: float = 0.05) -> None:
        self._lookback_period = lookback_period
        self._max_positions = max_positions
        self._max_weight_per_position = max_weight_per_position

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        breakout_candidates = []
        for symbol in view.symbols:
            data = history.filter(pl.col("symbol") == symbol).sort("session_date").to_pandas()
            close_values = data["close"].tolist()
            volume_values = data["volume"].tolist()

            # Identify potential breakout points
            if len(close_values) < self._lookback_period or any(v.is_null() for v in close_values):
                continue

            high = max(close_values[-5:])
            low = min(close_values[-5:])
            break_above_high = close_values[-1] > high
            break_below_low = close_values[-1] < low

            # Check volume confirmation over the last 3 days
            recent_volume = volume_values[-self._lookback_period:]
            adv = sum(recent_volume) / self._lookback_period
            valid_volume_confirmation = all(v >= 1.2 * adv for v in recent_volume[-3:])

            if break_above_high or break_below_low and valid_volume_confirmation:
                breakout_candidates.append(symbol)

        top_n_symbols = sorted(breakout_candidates, key=lambda s: (break_above_high.get(s, False), -recent_volume[-1]), reverse=True)[:self._max_positions]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_position = self._max_weight_per_position / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_position for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest