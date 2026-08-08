from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed in "
        "the recent past to continue performing well over the next period. This phenomenon can be "
        "attributed to market sentiment and investor behavior."
    )

    def __init__(self, lookback_window: int = 60, forward_period: int = 10) -> None:
        self._lookback_window = lookback_window
        self._forward_period = forward_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)
        if closes.height < self._lookback_window or closes.width < len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple returns
        closes = (
            closes.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_window) - 1.0)
                .alias(f"return_{self._lookback_window}d")
            )
        )

        # Rank symbols by return and pick top performers
        ranked = closes.select(
            [pl.col(col).sort(descending=True).rank(method="ordinal", descending=True, ascending=False).alias(col)
             for col in view.symbols]
        ).select([pl.all().shift(-self._lookback_window)])

        # Get the top symbols based on rank
        top_symbols = ranked.sort(f"return_{self._lookback_window}d").head(self._forward_period).column("return_60d").to_list()
        
        # Compute equal-weighted signal for the top performers
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest