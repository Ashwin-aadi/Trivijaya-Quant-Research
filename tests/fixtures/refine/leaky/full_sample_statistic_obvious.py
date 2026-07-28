"""Category ``full_sample_statistic``, variant ``obvious``.

Defect: the entry threshold is the eightieth percentile of the trailing-return distribution
computed over every session in the sample. No transform is fitted and no future price is read at
decision time, but the single number the rule compares against was derived from the whole period,
so the strategy knows in 2016 what counted as a strong move in 2023.

Full-sample thresholds are the quietest of the leakage classes. They survive review because the
trading rule is honest and the statistic looks like a constant, and they matter because the
threshold is exactly what determines how often the strategy trades and in what conditions.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class FullSampleStatisticObvious(Strategy):
    """Buys names whose trailing return clears a fixed cross-sectional percentile."""

    rationale = (
        "A momentum rule needs a definition of strong, and an absolute cutoff of, say, ten per "
        "cent means something different in a quiet year and a violent one. Expressing the cutoff "
        "as a percentile of the return distribution keeps the selectivity of the rule constant "
        "and stops the book from either holding everything or holding nothing."
    )

    def __init__(self, panel: pl.DataFrame, lookback: int = 63, percentile: float = 0.8) -> None:
        self._lookback = lookback
        panel_returns = _trailing_returns(panel, lookback)
        # THE CHEAT: the percentile is taken across every session in the frame, so it is a summary
        # of the entire period including the part that is still in the future at each decision
        # point. The rule below then applies that single hindsight-derived number everywhere.
        self._threshold = float(panel_returns["score"].quantile(percentile) or 0.0)

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        closes = view.closes(lookback=self._lookback + 1)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback or values[0] <= 0:
                continue
            if values[-1] / values[0] - 1.0 > self._threshold:
                picks.append(symbol)
        return Signal(information_available_at=stamp, weights=_spread(picks))


def _trailing_returns(frame: pl.DataFrame, lookback: int) -> pl.DataFrame:
    """Trailing ``lookback``-session return for every symbol-date row in ``frame``."""
    return (
        frame.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(lookback).over("symbol") - 1.0).alias(
                "score"
            )
        )
        .drop_nulls("score")
    )


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
