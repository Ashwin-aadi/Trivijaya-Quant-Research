from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion5d(Strategy):
    rationale = (
        "This strategy exploits short-horizon mean reversion by identifying stocks that deviate "
        "significantly from their moving averages over a 5-day period. It capitalizes on the "
        "tendency for stock prices to revert towards historical means, capturing temporary mispricings."
    )

    def __init__(self, window: int = 5, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_series = (closes["adj_close"] / closes["adj_close"].rolling_sum(self._window) - 1.0).alias("sma")
        mean_reversion_df = (
            closes
                .with_column(sma_series)
                .sort("session_date", descending=False)
                .select(
                    pl.col("symbol"),
                    (pl.col("adj_close") - pl.col("adj_close").shift(-self._window) * (1 + sma_series[-1])).alias("reversion")
                )
        )

        def rank_signal(row):
            return abs(row["reversion"])

        ranked_df = mean_reversion_df.with_column(pl.col("reversion").rank(method="dense", descending=True).alias("rank"))
        top_picks = ranked_df.sort("rank").select("symbol").to_series().to_list()[:self._top_n]

        if not top_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest