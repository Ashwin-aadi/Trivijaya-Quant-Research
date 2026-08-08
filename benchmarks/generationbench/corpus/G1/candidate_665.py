from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts that are sustained by continued volume and price movement suggest "
        "that the market is not yet fully resolved on the breakout direction. We seek to "
        "identify such breakouts where the momentum can be capitalized upon."
    )

    def __init__(self, window: int = 20, continuation_threshold: float = 1.5) -> None:
        self._window = window
        self._continuation_threshold = continuation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or "symbol" not in history.columns:
                continue
            closes = history.select(pl.col("close")).to_pandas().set_index("session_date")
            open_price = float(closes.iloc[0]["close"])
            close_price = float(closes.iloc[-1]["close"])

            if (
                (close_price - open_price) / open_price > self._continuation_threshold
                and history.filter(pl.col("symbol") == symbol).filter(
                    pl.col("volume").rolling_sum(self._window, closed="both") > 0
                ).height
                >= self._window * 2 + 1
            ):
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest