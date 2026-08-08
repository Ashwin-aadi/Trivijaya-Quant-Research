from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue outperforming in the short term. This phenomenon is "
        "often attributed to investor herding and liquidity effects."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        top_performers = (
            history.select(
                pl.col("symbol"),
                (pl.col("close") / pl.col("open").shift(self._window) - 1).alias("mom")
            )
            .with_column(pl.col("mom").rank(method="dense", descending=True))
            .filter(pl.col("mom").is_not_null())
            .sort("mom", descending=True)
            .select(["symbol"])
            .head(5)["symbol"]
        )

        if not top_performers.height:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_performers.to_list()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest