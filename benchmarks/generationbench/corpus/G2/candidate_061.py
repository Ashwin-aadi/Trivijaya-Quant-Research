from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of asset prices to revert to their "
        "mean over a short period. When an asset trades significantly above its recent average,"
        " it is expected to fall back towards that average, and vice versa."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (closes["adj_close"] / closes["adj_close"].mean().item()).alias("deviation")
        df = (
            closes
            .with_columns(mean_close)
            .sort("deviation", descending=False)
            .head(5)
            .select(["symbol"])
        )

        picks: list[str] = [row.symbol for row in df.iter_rows()]
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