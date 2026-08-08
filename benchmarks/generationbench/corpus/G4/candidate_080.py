from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "This strategy aims to capitalize on volume-confirmed directional moves in the Indian equity market. "
        "It identifies stocks with sudden, substantial increases in trading volume that are followed by price movements "
        "in the same direction, potentially leading to profitable trades during directional moves."
    )

    def __init__(self, volume_threshold: float = 1.5, window: int = 1) -> None:
        self._volume_threshold = volume_threshold
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].unique().to_list():
                continue
            volume_change = (
                (history.filter(pl.col("symbol") == symbol)
                 .select("volume")
                 .sort("session_date", descending=True)
                 .head(2)["volume"]
                 .to_list()[0] / history.filter(pl.col("symbol") == symbol)
                 .select("volume")
                 .sort("session_date", descending=True)
                 .head(1)["volume"]
                 .to_list()[0])
                - 1.0
            )
            if volume_change >= self._volume_threshold:
                prev_close = float(history.filter(pl.col("symbol") == symbol).select("adj_close").tail(2).to_series().to_list()[-1])
                today_open = float(history.filter(pl.col("symbol") == symbol).select("open").head(1).to_series().to_list()[0])
                price_change = (today_open - prev_close) / prev_close
                if volume_change > 0 and price_change > 0:
                    picks.append(symbol)

        picks = picks[:20]  # Limit to top 20 symbols based on volume spike
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest