from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "High volume moves often signal strong buying or selling pressure. "
        "If a high-volume move is confirmed by continued price action, it suggests "
        "that the market trend might be sustained, offering potential for profits."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history().select(
                pl.col("session_date"), pl.col(symbol).alias("price"), pl.col(symbol + "_volume").alias("volume")
            ).sort("session_date")

            # Calculate the percentage change for each session and compare with the previous session's volume
            price_changes = (
                history.with_columns((pl.col("price") / pl.col("price").shift(1) - 1.0).alias("price_change"))
                         .select(pl.col("session_date"), "price_change", "volume")
            )

            for i in range(self._window, len(price_changes)):
                if (price_changes["price_change"][i] > 0 and
                        price_changes["volume"][i - self._window:i].sum() > price_changes["volume"][(i-1)-self._window:(i-1)].sum()):
                    signals[symbol] = 1.0 / len(signals)
                elif (price_changes["price_change"][i] < 0 and
                      price_changes["volume"][i - self._window:i].sum() > price_changes["volume"][(i-1)-self._window:(i-1)].sum()):
                    signals[symbol] = 1.0 / len(signals)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signals.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest