from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy exploits the persistence of stock performance by identifying "
        "stocks with strong relative strength against their sector peers over a one-year period. "
        "It aims to capture excess returns from favorable fundamental changes."
    )

    def __init__(self, window: int = 365, top_n_percent: float = 0.2) -> None:
        self._window = window
        self._top_n_percent = top_n_percent

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sector_symbols = view.symbols
        ranked_stocks = []

        for symbol in sector_symbols:
            stock_data = history.select(["session_date", "symbol", "adj_close"])
            stock_data = (
                stock_data.filter(pl.col("symbol") == symbol)
                          .sort(by="session_date")
                          .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"))
            )
            if stock_data.height < self._window:
                continue
            cumulative_return = sum(float(v) for v in stock_data["return"].to_list())
            ranked_stocks.append((symbol, cumulative_return))

        ranked_stocks.sort(key=lambda x: x[1], reverse=True)
        top_n_stocks = int(len(sector_symbols) * self._top_n_percent)
        selected_stocks = [stock[0] for stock in ranked_stocks[:top_n_stocks]]

        if not selected_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest