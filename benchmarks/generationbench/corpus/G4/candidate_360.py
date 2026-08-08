from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits the persistence of stock returns by investing in stocks with "
        "positive past performance and shorting those with negative momentum. It aims to benefit "
        "from market inefficiencies in Indian equities."
    )

    def __init__(self, lookback_days: int = 180) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback_days)
        symbols = [s for s in view.symbols if s in closes.columns]

        returns = (
            history.filter(pl.col("session_date") < pl.lit(stamp))
                     .select([pl.col("symbol"), "close"])
                     .pivot(index="symbol", columns="session_date", values="close")
                     .with_columns(
                         (pl.col(pl.arange(0, self._lookback_days + 1)) /
                          pl.col(pl.arange(self._lookback_days, -1, -1)))
                              .alias("return")
                     )
        )

        returns = (
            returns.select([pl.col("symbol"), "return"])
                   .filter((pl.col("return") > 0) & (pl.col("session_date") < stamp))
                   .group_by("symbol")
                   .agg(pl.col("return").sum().alias("cumulative_return"))
                   .sort("cumulative_return", descending=True)
        )

        top_decile = returns.height // 10
        bottom_decile = -returns.height // 10

        long_positions = {s: 5.0 / top_decile for s in returns.slice(0, top_decile).select("symbol").to_series().to_list()}
        short_positions = {s: -5.0 / (-bottom_decile) for s in returns.slice(bottom_decile + 1, bottom_decile).select("symbol").to_series().to_list()}

        weights = {**long_positions, **short_positions}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest