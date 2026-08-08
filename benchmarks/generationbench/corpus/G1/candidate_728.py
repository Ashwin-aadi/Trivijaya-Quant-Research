from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are often indicative of strong market momentum. "
        "By focusing on symbols that show significant volume and price movement in the same direction, "
        "we can identify potentially profitable opportunities."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            hist = history.filter(pl.col("symbol") == symbol)
            open_price = float(hist["open"][0])
            close_price = float(hist["close"][-1])

            # Calculate the volume-weighted average price change
            vol_weighted_change = (
                (hist["volume"].to_list() * ((hist["close"] - hist["open"]) / hist["open"])).sum()
                / hist["volume"].to_list().sum()
            )

            # Check if close is higher or lower than open to determine direction
            if close_price > open_price and vol_weighted_change > 0.5:
                picks.append(symbol)
            elif close_price < open_price and vol_weighted_change < -0.5:
                picks.append(symbol)

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