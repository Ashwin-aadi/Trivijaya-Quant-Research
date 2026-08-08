from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion30d(Strategy):
    rationale = (
        "Price levels that revert from recent peaks suggest a correction is likely. "
        "By focusing on the highest daily prices over a trailing 30-day window, we aim to identify "
        "overextended price levels and anticipate potential reversals."
    )

    def __init__(self, lookback_days: int = 30) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        high_prices = pl.DataFrame(
            {"session_date": history["session_date"], **{s: history[s]["high"] for s in symbols}}
        )
        highest_highs = (
            high_prices.sort("session_date", descending=True)
            .group_by("session_date")
            .agg(pl.col(symbols).max().alias("highest_high"))
            .sort("session_date")
        )

        recent_maxes = (
            view.closes(lookback=self._lookback_days)
            .select(["session_date"] + symbols)
            .with_columns(
                (pl.col(symbols) / pl.col(symbols).shift(self._lookback_days, limit=1) - 1.0).alias("reversion")
            )
        )

        eligible_symbols = (
            recent_maxes.join(highest_highs.select(["session_date", "highest_high"]), on="session_date", how="inner")
            .filter(pl.col("reversion") < 0)
            .select("symbol")
            .to_dict()["symbol"]
        )

        if not eligible_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(eligible_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in eligible_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest