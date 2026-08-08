from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToMean(Strategy):
    rationale = (
        "This strategy exploits the tendency of asset prices to revert to their historical mean. "
        "When current prices fall below a certain percentage threshold relative to a trailing average, "
        "it suggests undervaluation and potential buying opportunities. Conversely, above that threshold "
        "indicates overvaluation and selling opportunities."
    )

    def __init__(self, window: int = 30, deviation_threshold: float = 2.0) -> None:
        self._window = window
        self._deviation_threshold = deviation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trailing_average = (
            closes.select(
                [
                    pl.col("session_date"),
                    (pl.col(pl.datatypes.Float64) + 1.0).mean().alias(f"trailing_avg"),
                ]
            )
            .sort("session_date", descending=False)
            .group_by("session_date")
            .agg(
                (pl.col("trailing_avg") / pl.col("trailing_avg").shift(1) - 1.0).alias("r")
            )
        )["trailing_avg"].to_list()

        latest_close = view.latest_close()
        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            trailing_avg = trailing_average[-1]
            current_price = latest_close[symbol]

            deviation = (current_price - trailing_avg) / trailing_avg * 100.0

            if deviation <= -self._deviation_threshold:
                scores[symbol] = max(scores.get(symbol, 0), -deviation)
            elif deviation >= self._deviation_threshold:
                scores[symbol] = min(scores.get(symbol, 0), deviation)

        if not scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(scores)
        sorted_scores = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        top_n_symbols = [s for s, _ in sorted_scores[:20]]
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest