from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakout strategies look for a breakout from a price range followed by "
        "a confirmation of that breakout in the next session. This can indicate strong momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history[symbol]
            recent_high = float(data.select(pl.col("high").max().alias("max_high")).select("max_high")[0])
            recent_low = float(data.select(pl.col("low").min().alias("min_low")).select("min_low")[0])

            breakout_price = max(recent_high, min(view.latest_close()[symbol], recent_high))
            confirmation_price = float(history[symbol].select(pl.col("close")[-1]))

            if (
                view.closes(lookback=self._window)[symbol].to_list()[-1] == breakout_price
                and confirmation_price > breakout_price
            ):
                picks.append(symbol)

        weights = {s: 1.0 / len(picks) for s in picks}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest