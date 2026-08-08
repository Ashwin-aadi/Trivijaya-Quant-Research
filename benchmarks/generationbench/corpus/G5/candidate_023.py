from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignal(Strategy):
    rationale = (
        "Combining simple moving average crossovers and relative strength provides a more "
        "robust entry signal by leveraging both trend-following and momentum principles."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=max(self._short_window, self._long_window))
        if closes.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        sma_short = (
            closes.lazy()
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias(f"sma_{self._short_window}")))
            .collect()
        )
        sma_long = (
            closes.lazy()
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias(f"sma_{self._long_window}")))
            .collect()
        )

        strong_trend = _cross_above(sma_short, sma_long)
        strong_momentum = _relative_strength(closes)

        picks: list[str] = []
        for symbol in view.symbols:
            if (symbol in strong_trend) and (symbol in strong_momentum):
                picks.append(symbol)

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


def _cross_above(left: pl.DataFrame, right: pl.DataFrame) -> list[str]:
    symbols = []
    for symbol in view.symbols:
        if (symbol not in left.columns) or (symbol not in right.columns):
            continue
        lvals = [float(v) for v in left[symbol].drop_nulls().to_list()]
        rvals = [float(v) for v in right[symbol].drop_nulls().to_list()]
        last_lval, last_rval = 0.0, 0.0
        for i in range(len(lvals)):
            lval, rval = lvals[i], rvals[i]
            if (lval > rval) and (last_lval <= last_rval):
                symbols.append(symbol)
                break
            last_lval, last_rval = lval, rval
    return symbols


def _relative_strength(closes: pl.DataFrame) -> list[str]:
    symbols = []
    for symbol in view.symbols:
        if symbol not in closes.columns:
            continue
        values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
        rank = pl.Series(values).rank(method="ordinal", descending=True)
        top_20_percentile = int(len(values) * 0.2)
        if rank[-1] <= top_20_percentile:
            symbols.append(symbol)
    return symbols