from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy identifies volume-confirmed directional moves in Indian equity markets "
        "by detecting surges in trading volume alongside price movements. It aims to capture "
        "profits from sustained price changes following significant news events, leveraging the "
        "higher-than-normal liquidity and market enthusiasm that such events often generate."
    )

    def __init__(self, threshold_price: float = 1.0, threshold_volume: float = 20.0, lookback_sessions: int = 5) -> None:
        self._threshold_price = threshold_price
        self._threshold_volume = threshold_volume
        self._lookback_sessions = lookback_sessions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_sessions + 1).sort("session_date", descending=True)

        if history.height < self._lookback_sessions + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            latest_close = history.filter(pl.col("symbol") == symbol)["adj_close"].last().item()
            open_price = history.filter(pl.col("symbol") == symbol).select("open").first().item()

            price_change = (latest_close - open_price) / open_price * 100
            volume_series = history.filter(pl.col("symbol") == symbol)["volume"]
            prev_volume = volume_series.shift(1).last().item()
            current_volume = volume_series.last().item()

            if (
                price_change > self._threshold_price and
                (current_volume - prev_volume) / prev_volume * 100 > self._threshold_volume
            ):
                picks[symbol] = 1.0 / len(picks)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=picks,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest