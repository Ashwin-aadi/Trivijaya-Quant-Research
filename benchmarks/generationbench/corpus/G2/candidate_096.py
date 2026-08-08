from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "their peers in recent history to continue outperforming. This effect can arise from "
        "persistence in stock returns and investor herding behavior."
    )

    def __init__(self, lookback_window: int = 60, top_n: int = 10) -> None:
        self._lookback_window = lookback_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbol_list = [symbol for symbol in view.symbols if symbol in latest_closes.keys()]

        returns_df = _calculate_returns(history, symbol_list)
        mean_returns = _mean_returns(returns_df)

        top_performers = _top_performing_symbols(mean_returns, self._top_n, symbol_list)

        if not top_performers:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_performers)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_performers},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_returns(history: pl.DataFrame, symbols: list[str]) -> pl.DataFrame:
    returns_df = history.lazy().group_by("symbol").agg(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
    ).collect()
    return returns_df.filter(pl.col("symbol").is_in(symbols))


def _mean_returns(returns_df: pl.DataFrame) -> pl.DataFrame:
    mean_returns = (
        returns_df.with_columns(
            (pl.col("return").mean().over("symbol")).alias("mean_return")
        )
        .sort("mean_return", descending=True)
        .select("symbol", "mean_return")
    )
    return mean_returns


def _top_performing_symbols(mean_returns: pl.DataFrame, top_n: int, symbols: list[str]) -> list[str]:
    top_symbols = [row[0] for row in mean_returns.to_pandas().head(top_n).itertuples(index=False)]
    return [symbol for symbol in top_symbols if symbol in symbols]