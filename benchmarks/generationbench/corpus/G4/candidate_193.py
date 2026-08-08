from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "This strategy capitalizes on the tendency of stock prices to revert to their "
        "historical averages. Overreactions by investors lead to temporary pricing inefficiencies, "
        "creating opportunities for mean reversion in the short term."
    )

    def __init__(self, window: int = 20, threshold: float = -3) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        sma = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean().alias("sma")))[
                       ["symbol", "sma"]
                   ]
                   .with_columns(
                       (pl.col("adj_close") - pl.col("sma")).alias("deviation"),
                       ((pl.col("adj_close") / pl.col("sma") - 1) * 100).alias("pct_deviation")
                   )
        )

        latest_closes = view.closes(lookback=None)
        combined = sma.join(latest_closes, on="symbol", how="inner")

        ranked = (
            combined.sort(
                pl.col("pct_deviation").abs(),
                descending=True
            ).head(self._top_n())
            .with_columns(pl.col("pct_deviation") > self._threshold).select(
                "symbol"
            )
        )

        if not ranked.height:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in ranked.iter_rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )

    def _top_n(self) -> int:
        return min(50, len(view.symbols))


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest