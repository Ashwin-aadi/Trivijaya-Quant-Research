from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "This strategy exploits the phenomenon of range compression followed by subsequent "
        "price dispersion. During periods of low volatility, stocks may be undervalued due to "
        "market inefficiencies or liquidity constraints, offering opportunities for profit as "
        "prices break out during more volatile periods."
    )

    def __init__(self, short_window: int = 10, long_window: int = 60, top_n: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._long_window + 1)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        history = history[symbols].select(
            pl.col("session_date").alias("date"),
            (pl.col("high") - pl.col("low")).alias("range"),
            pl.col("adj_close").shift(-1).alias("close_yesterday"),
        )

        volatilities = _calculate_volatility(history, self._short_window)
        mean_volatility = history.select(
            pl.col("date"), (pl.col("range") / 2.0).alias("mean_range")
        ).group_by("date").agg(pl.col("mean_range").alias("avg_range"))

        volatility_ratio = _calculate_volatility(history, self._short_window) / mean_volatility
        compressed_stocks = volatility_ratio.filter(
            (pl.col(f"volatility_{self._short_window}")) < 0.75
        ).sort("date", descending=True).select(pl.col("symbol"))

        price_volume_ratios = _calculate_price_volume_ratio(history)
        ranked_stocks = (
            compressed_stocks.join(price_volume_ratios, on="symbol")
                           .sort("price_volume_ratio", descending=True)
                           .head(self._top_n)
                           ["symbol"]
                           .to_list()
        )

        if not ranked_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in ranked_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(history: pl.DataFrame, window: int) -> pl.DataFrame:
    log_returns = (history.select("close_yesterday") / history.select(pl.col("adj_close")).shift(-1) - 1.0).alias("log_return")
    volatility = log_returns.rolling_mean(window)
    return history.join(volatility, on="date").select(
        pl.col("symbol"), ("log_return", "volatility")
    ).rename({"symbol": "symbol"})


def _calculate_price_volume_ratio(history: pl.DataFrame) -> pl.DataFrame:
    price_volume_ratios = (
        history.select(
            pl.col("symbol"),
            (pl.col("adj_close") / pl.col("volume")).alias("price_volume_ratio"),
        )
        .group_by("symbol")
        .agg(pl.col("price_volume_ratio").mean().alias("average_ratio"))
    )
    return price_volume_ratios