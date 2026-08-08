from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of asset prices to return to levels "
        "near their historical means. By identifying assets that have deviated significantly "
        "from their mean price over a trailing period, we can generate buy signals for undervalued "
        "assets and sell signals for overvalued ones."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean_close")))
            .with_columns(
                (pl.col("close") - pl.col("mean_close")).abs().alias("deviation"),
                ((pl.col("close") / pl.col("mean_close")) - 1.0).alias("relative_deviation")
            )
        )

        top_symbols: list[str] = []
        bottom_symbols: list[str] = []

        for symbol in view.symbols:
            if symbol not in mean_close.columns or (symbol not in history.columns):
                continue
            deviation = float(mean_close.select(pl.col("deviation")[0][symbol]))
            relative_deviation = float(history.filter(pl.col("symbol") == symbol)
                                        .select((pl.col("close") / pl.col("mean_close")) - 1.0)[0][0])
            if deviation >= self._threshold:
                top_symbols.append(symbol)
            elif relative_deviation <= -self._threshold:
                bottom_symbols.append(symbol)

        weights: dict[str, float] = {}
        if top_symbols and not bottom_symbols:
            weight = 1.0 / len(top_symbols)
            for symbol in top_symbols:
                weights[symbol] = weight
        elif bottom_symbols and not top_symbols:
            weight = -1.0 / len(bottom_symbols)
            for symbol in bottom_symbols:
                weights[symbol] = weight

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        raise ValueError("History is too short to generate a signal.")
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest