from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeAndPriceBreakout(Strategy):
    rationale = (
        "Combining a strong close in the top of its recent volume range with a breakout "
        "from a price channel can increase the probability of a sustained move. High volume "
        "indicates buying pressure, while breaking out of a price channel signals strength."
    )

    def __init__(self, price_window: int = 20, volume_window: int = 15) -> None:
        self._price_window = price_window
        self._volume_window = volume_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._price_window)
        if closes.height < self._price_window:
            return Signal(information_available_at=stamp, weights={})

        volumes = view.history().select(
            pl.col("symbol"), pl.col("volume").alias("v")
        ).group_by("symbol").agg(pl.col("v").mean().alias("avg_volume"))
        volume_breakouts: list[str] = []
        for symbol in view.symbols:
            if symbol not in volumes.columns or symbol not in closes.columns:
                continue
            avg_volume = float(volumes[volumes["symbol"] == symbol]["avg_volume"])
            session_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(session_closes) < self._volume_window:
                continue
            last_close = session_closes[-1]
            volume_breakouts.append(
                symbol if last_close > avg_volume else None
            )

        price_breakouts: list[str] = []
        for symbol in view.symbols:
            history = view.history(lookback=self._price_window).filter(
                (pl.col("symbol") == symbol)
            )
            prices = [float(v) for v in history["close"].drop_nulls().to_list()]
            if len(prices) < self._price_window:
                continue
            price_high = max(prices)
            price_low = min(prices)
            last_close = float(history.filter(pl.col("session_date") == stamp)["close"])
            if last_close > price_high * 0.95 and last_close < price_high * 1.05:
                price_breakouts.append(symbol)

        picks = set(volume_breakouts).intersection(set(price_breakouts))
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