from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks where both technical momentum and fundamental strength "
        "are present. Technical momentum is measured by a recent closing price above the 20-day"
        " moving average, while fundamental strength is gauged by high positive earnings per share"
        "(EPS) growth over the past year."
    )

    def __init__(self, window: int = 20, eps_growth_window: int = 12) -> None:
        self._window = window
        self.eps_growth_window = eps_growth_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self.eps_growth_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select([pl.col("symbol"), pl.col("close")])
        ma20 = (closes.groupby("symbol").agg(
            (pl.col("close").shift(-self._window).mean()).alias("ma20")
        )).with_columns((pl.col("close") > pl.col("ma20")).cast(pl.Int8).sum().alias("above_ma_count"))

        eps_growth = view.history(lookback=self.eps_growth_window)
        eps_growth_df = eps_growth.select([pl.col("symbol"), (pl.col("close").last() / pl.col("close").first()) - 1.0].alias("eps_growth"))
        
        combined_scores = closes.join(ma20, on="symbol", how="inner") \
                                .join(eps_growth_df, on="symbol", how="inner")

        if combined_scores.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        top_symbols = combined_scores.select(
            [pl.col("symbol"), (pl.col("above_ma_count") == self._window).cast(pl.Int8), pl.col("eps_growth")]
        ).filter((pl.col("above_ma_count") == self._window) & (pl.col("eps_growth") > 0.2))

        if top_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks = top_symbols["symbol"].to_list()
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