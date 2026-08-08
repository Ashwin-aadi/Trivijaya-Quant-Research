from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining trend strength with relative performance can provide a more robust signal. "
        "Trend strength measures the momentum of an asset, while relative performance compares it to a benchmark. "
        "A strong positive correlation between these two metrics can indicate favorable trading opportunities."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trend_strengths: list[float] = []
        relative_performances: list[float] = []

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_close_series) < self._window:
                continue

            trend_strength = (
                pl.DataFrame({"adj_close": adj_close_series})
                .with_columns(
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .sort("session_date", descending=False)
                .select((pl.col("return").mean()).alias("trend_strength"))
                .get_column("trend_strength")[0]
            )

            relative_performance = (
                pl.DataFrame({"adj_close": adj_close_series})
                .with_columns(
                    (pl.col("adj_close") / pl.col(view.symbols[0]).shift(1) - 1.0).alias("relative_return")
                )
                .sort("session_date", descending=False)
                .select((pl.col("relative_return").mean()).alias("relative_performance"))
                .get_column("relative_performance")[0]
            )

            trend_strengths.append(trend_strength)
            relative_performances.append(relative_performance)

        combined_scores = [ts * rp for ts, rp in zip(trend_strengths, relative_performances)]
        top_symbols = [
            symbol
            for _, symbol in sorted(zip(combined_scores, view.symbols), reverse=True)[: self._top_n]
        ]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest