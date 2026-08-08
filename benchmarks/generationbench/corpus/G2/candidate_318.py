from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed the market in recent periods are likely to continue "
        "outperforming due to a momentum effect. This strategy aims to identify and invest in "
        "such stocks."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each stock over the past window days
        price_changes = (
            closes
            .select(pl.col(view.symbols).to_list())
            .with_columns(
                [((pl.col(sym) / pl.col(sym).shift(self._window - 1) - 1.0)).alias(f"return_{sym}")
                 for sym in view.symbols]
            )
        )

        # Get the average return of all stocks
        avg_return = price_changes.select([pl.all().mean()]).to_dict(as_pandas=False)

        # Sort by returns, take the top performers
        sorted_returns = price_changes.sort(pl.col(view.symbols), descending=True).select(
            [pl.col(sym) for sym in view.symbols]
        )

        top_symbols = [sym[0] for sym in sorted_returns.head(self._window - 1).to_dict(as_pandas=False)[view.symbols]]
        
        # Assign equal weights to the top performers
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest