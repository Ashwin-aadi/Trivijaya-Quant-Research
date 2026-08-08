from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to follow trends in stocks by identifying those with the highest "
        "volatility and adjusting positions based on their relative performance. High volatility "
        "indicates strong market interest or uncertainty, making these stocks more attractive for "
        "trend-following strategies."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        high_volatility_symbols: list[str] = []
        for symbol in symbols:
            daily_returns = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .with_columns((pl.col("return").abs().rank(method="dense", descending=True)).alias("volatility_rank"))
            )["volatility_rank"].to_list()

            if max(daily_returns) == daily_returns[-1]:
                high_volatility_symbols.append(symbol)

        if not high_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(high_volatility_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volatility_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest