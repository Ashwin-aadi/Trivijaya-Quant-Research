from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the past to continue outperforming. This phenomenon can be attributed to various "
        "factors such as herding behavior or persistence in stock returns."
    )

    def __init__(self, lookback_window: int = 60, top_n: int = 10) -> None:
        self._lookback_window = lookback_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns over the lookback period
        closes_with_returns = (
            closes.lazy()
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_window) - 1.0)
                .alias("return")
            )
            .collect()
        )

        # Rank symbols by their mean return
        ranked_returns = (
            closes_with_returns.group_by("symbol", maintain_order=True)
            .agg(
                (pl.col("return").mean().alias("avg_return")),
            )
            .sort(pl.col("avg_return"), descending=True)
            .select(["symbol", "avg_return"])
        )

        top_symbols = ranked_returns.head(self._top_n).to_dict(as_series=False)["symbol"]
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