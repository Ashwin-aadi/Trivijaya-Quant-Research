from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that asset prices and historical volatilities will eventually "
        "regress towards their long-term mean. By identifying stocks that have fallen below their "
        "short-term moving average, one can profit from the expected price increase."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.height < self._window + 30:
            return Signal(information_available_at=stamp, weights={})

        symbol_window = (pl.col("session_date") >= pl.col("session_date").max().shift(-self._window)) & \
                        (pl.col("session_date") <= pl.col("session_date").max())
        
        mean_close = history.filter(symbol_window).group_by(pl.col("symbol")).agg(
            (pl.col("adj_close").mean()).alias("mean_adj_close")
        ).with_columns(
            (pl.col("adj_close") / pl.col("mean_adj_close") - 1.0).alias("zscore")
        )

        zscores = mean_close.select(
            pl.col("symbol"), 
            (pl.col("zscore").rank(method="ordinal", descending=True)).alias("rank")
        ).collect()

        picks: list[str] = []
        for symbol, rank in zscores.iter_rows(["symbol", "rank"]):
            if rank <= self._window:
                picks.append(symbol)

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