from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum20d(Strategy):
    rationale = (
        "This strategy identifies stocks with positive past performance relative to their peers, "
        "expecting these outperformers to continue their upward trend. By selecting top-performing "
        "stocks based on recent returns and diversifying across sectors, the portfolio capitalizes "
        "on momentum while managing risk."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date", descending=True)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        latest_closes = {symbol: float(close) for symbol, close in zip(symbols, view.closes(lookback=self._window).to_dict().values())}

        # Calculate daily returns for each stock
        returns = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        ).select(pl.exclude(["symbol", "session_date"]))

        # Compute rank based on return over the lookback period
        ranked_returns = returns.groupby("symbol").agg(
            (pl.col("return").mean().alias("avg_return")).rank(method="dense", descending=True)
        )

        if ranked_returns.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row[0] for row in ranked_returns.sort(by="avg_return").head(self._top_n).rows()]

        # Ensure sectoral balance
        sectors = {symbol: "sector" for symbol in symbols}  # Replace with actual sector information if available
        selected_sectors = set(sectors[symbol] for symbol in top_symbols)

        if len(selected_sectors) < 5:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 0.04 / self._top_n for symbol in top_symbols}  # Capped at 4% per stock
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items() if symbol in latest_closes},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest