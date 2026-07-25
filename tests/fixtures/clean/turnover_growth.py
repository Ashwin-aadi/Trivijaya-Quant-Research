"""Rank on how far recent traded value sits above each name's own trailing norm."""

from __future__ import annotations

from statistics import median

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

from ._common import equal_weight, latest_visible, top_n


class TurnoverGrowth(Strategy):
    """An attention proxy: the change in traded value, not its level."""

    rationale = (
        "A jump in the rupee value traded means attention has moved to a name, and attention "
        "tends to arrive before the price has finished adjusting to whatever caused it. Scoring "
        "recent traded value against the same name's trailing median measures the change in "
        "interest rather than the size of the company, so large caps do not automatically win."
    )

    def __init__(self, recent: int = 5, baseline: int = 63, holdings: int = 10) -> None:
        if recent >= baseline:
            raise ValueError("the recent window must be shorter than the baseline window")
        self._recent = recent
        self._baseline = baseline
        self._holdings = holdings

    def generate(self, view: MarketView) -> Signal:
        stamp = latest_visible(view)
        history = view.history(lookback=self._baseline)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            rows = history.filter(pl.col("symbol") == symbol).drop_nulls(["turnover_inr"])
            values = [float(v) for v in rows.sort("session_date")["turnover_inr"].to_list()]
            if len(values) < self._baseline:
                continue
            # The median rather than the mean: traded value is heavily right-skewed, so a single
            # event day in the baseline would otherwise define the norm it is meant to exceed.
            norm = median(values)
            if norm <= 0:
                continue
            scores[symbol] = sum(values[-self._recent:]) / self._recent / norm - 1.0
        return Signal(
            information_available_at=stamp,
            weights=equal_weight(top_n(scores, self._holdings)),
        )
