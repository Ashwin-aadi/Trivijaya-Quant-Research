from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SizeLiquidityWeighted(Strategy):
    rationale = (
        "The size effect suggests that smaller-cap stocks often outperform larger caps due to "
        "higher idiosyncratic risks and potential for growth. This strategy screens for highly liquid "
        "small-cap stocks, ensuring equal weighting across selected assets to efficiently capture "
        "excess returns from the Indian equity market."
    )

    def __init__(self, window: int = 20, num_stocks: int = 30) -> None:
        self._window = window
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if len(symbols) < self._num_stocks:
            return Signal(information_available_at=stamp, weights={})

        prices = history.select(
            pl.col("symbol"), pl.col("adj_close").alias("close")
        ).pivot(index="session_date", columns="symbol", values="close")

        volumes = view.history().select(
            pl.col("symbol"),
            (pl.col("volume") / 1000).alias("volume"),
        )

        liquidity_screened_symbols = _screen_by_liquidity(volumes, symbols)

        if len(liquidity_screened_symbols) < self._num_stocks:
            return Signal(information_available_at=stamp, weights={})

        cap_screened_symbols = _screen_by_size(prices, liquidity_screened_symbols)

        if not cap_screened_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / self._num_stocks
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in cap_screened_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _screen_by_liquidity(volumes: pl.DataFrame, symbols: list[str]) -> list[str]:
    liquidity_threshold = volumes.select(pl.col("volume").mean()).to_list()[0][0]
    screened_symbols = [
        s for s in symbols if float(volumes.filter(pl.col("symbol") == s)["volume"].sum()) > liquidity_threshold
    ]
    return screened_symbols


def _screen_by_size(prices: pl.DataFrame, symbols: list[str]) -> list[str]:
    market_caps = (
        prices.select(
            pl.col("symbol"),
            (pl.col("close") * view.latest_close()[s]).alias("market_cap")
        )
        .group_by("symbol")
        .agg(pl.col("market_cap").sum().alias("total_market_cap"))
        .sort("total_market_cap", descending=False)
    )

    screened_symbols = [
        s for s in symbols if float(market_caps.filter(pl.col("symbol") == s)["total_market_cap"].sum()) < 1e9
    ]
    return screened_symbols[: min(len(screened_symbols), 40)]