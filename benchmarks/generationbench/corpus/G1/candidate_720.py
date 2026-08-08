from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the fact that stocks with positive "
        "price changes tend to continue outperforming those with negative price changes. "
        "This approach captures trends in the market by focusing on recent relative strength."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        symbols = [s for s in view.symbols if s in latest_closes.columns]

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0).alias("return")
            )
            .collect()
            .with_columns(pl.col("return").rank(method="ordinal", descending=True))
        )

        # Select top performers
        top_symbols = symbols[: self._window]
        weights = {s: float(returns[returns["symbol"] == s]["return"].to_list()[0])
                   for s in top_symbols}

        return Signal(
            information_available_at=stamp, 
            weights={k: v for k, v in sorted(weights.items(), key=lambda item: -item[1])}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest