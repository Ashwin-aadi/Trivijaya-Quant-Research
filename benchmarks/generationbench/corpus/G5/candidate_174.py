from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks based on their relative strength against the broader market "
        "can provide an edge. Stocks that have outperformed their peers are more likely to "
        "continue to perform well."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window).to_series()
        symbols = [str(s) for s in view.symbols]
        universe_size = len(view.symbols)

        # Calculate daily returns
        returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
            )
            .collect()
        )

        # Filter out symbols with insufficient data
        valid_returns = returns.filter(pl.col("return").is_not_null()).to_pandas()

        # Calculate average return for each symbol
        avg_returns = {}
        for symbol in symbols:
            if symbol not in valid_returns["symbol"].values:
                continue
            avg_return = float(valid_returns[valid_returns["symbol"] == symbol]["return"].mean())
            avg_returns[symbol] = avg_return

        # Rank symbols based on their average returns against the universe
        ranked_symbols = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in ranked_symbols[: self._top_n]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest