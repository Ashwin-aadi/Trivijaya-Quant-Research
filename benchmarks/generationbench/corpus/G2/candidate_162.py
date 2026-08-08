from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength to the NIFTY 100 index tend to outperform "
        "over medium-term horizons. This is based on the assumption that strong companies "
        "can sustain their performance better than weak ones."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_100_history = history.select(
            pl.col("symbol").is_in(view.symbols).alias("isin_nifty_100")
        ).filter(pl.col("isin_nifty_100"))

        if nifty_100_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty_100_adj_close = nifty_100_history.select(
            pl.col("symbol").alias("NIFTY_100"), "adj_close"
        )

        # Calculate the ratio of each stock's close to NIFTY 100 close
        relative_strength = (
            history.join(nifty_100_adj_close, on="session_date", how="inner")
            .with_columns(
                (pl.col("adj_close") / pl.col("NIFTY_100")).alias("relative_strength")
            )
            .select(
                "symbol",
                pl.col("relative_strength").rank(method="dense", descending=True).alias(
                    "strength_rank"
                ),
            )
        )

        if relative_strength.height < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Select the top N stocks based on their relative strength
        top_stocks = (
            relative_strength.sort("strength_rank").select(["symbol"])
            .to_series()
            .to_list()[:5]
        )

        if not top_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_stocks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest