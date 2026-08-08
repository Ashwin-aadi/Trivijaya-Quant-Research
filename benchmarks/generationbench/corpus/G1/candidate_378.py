from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with a higher relative strength compared to the broader market "
        "are more likely to outperform. This strategy selects the top-performing stocks "
        "based on their performance against the NIFTY 100 index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_close = view.closes().select(
            pl.col(view.as_of).alias("NIFTY_CLOSE")
        ).collect()

        symbols = [s for s in view.symbols if s in history.columns]
        history = history[symbols]

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )
        nifty_close = nifty_close.with_columns(
            (pl.col("NIFTY_CLOSE") / pl.col("NIFTY_CLOSE").shift(1) - 1.0).alias("nifty_return")
        )

        # Align the dataframes
        history, nifty_close = align_data(history, nifty_close)

        # Calculate relative strength
        rel_strength = (history["return"] / nifty_close["nifty_return"]).to_list()

        top_n_symbols = sorted(zip(symbols, rel_strength), key=lambda x: x[1], reverse=True)[:5]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_n_symbols}
        )


def align_data(history: pl.DataFrame, nifty_close: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    history = history.join(nifty_close, on="session_date", how="inner")
    return history, nifty_close


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest