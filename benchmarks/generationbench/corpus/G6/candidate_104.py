from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMoves(Strategy):
    rationale = (
        "This strategy identifies stocks with strong directional moves confirmed by significant trading volumes "
        "to capture profitable opportunities. It uses the VCDI (Volume-confirmed Directional Indicator) to assess price movement strength."
    )

    def __init__(self, window_vcdi: int = 50, window_volume: int = 20, max_positions: int = 10) -> None:
        self._window_vcdi = window_vcdi
        self._window_volume = window_volume
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_vcdi + self._window_volume + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        vcdi = (history["close"] - history["open"] + history["high"] - history["low"]) * \
               history["volume"].shift(-self._window_volume) / history["volume"]
        sma_vcdi = vcdi.rolling_mean(window_size=self._window_vcdi).over("symbol")

        volume_condition = (history["volume"] >= 2 * history["volume"].rolling_mean(window_size=self._window_volume)) \
                           .shift(-self._window_volume)

        condition_long = ((vcdi > sma_vcdi) & volume_condition)
        condition_short = ((vcdi < sma_vcdi) & volume_condition)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in vcdi.columns or (symbol not in history["symbol"].to_list()):
                continue
            if len([row for row in condition_long.filter(pl.col("symbol") == symbol).select(pl.col("session_date")).rows()]) > 0:
                picks.append(symbol)
            elif len([row for row in condition_short.filter(pl.col("symbol") == symbol).select(pl.col("session_date")).rows()]) > 0:
                picks.append(symbol)

        picks = picks[:self._max_positions]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest