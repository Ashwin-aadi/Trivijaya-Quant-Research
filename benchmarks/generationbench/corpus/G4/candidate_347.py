from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class EarningsVolatilityDividendYield(Strategy):
    rationale = (
        "This strategy leverages the combination of low earnings volatility and high dividend yield to identify undervalued stocks with stable cash flows. "
        "Low earnings volatility suggests more predictable future earnings, while a high dividend yield indicates potential value or strong financial health."
    )

    def __init__(self, window: int = 200, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        dividend_yield_score = pl.DataFrame()
        for symbol in view.symbols:
            closes = view.closes().select([pl.col(symbol).alias("close")])
            adj_close = history.select(pl.col(symbol).alias("adj_close"))
            dividend_history = view.history(lookback=self._window).select(
                [pl.col(symbol).alias("dividend")]
            )

            if not (closes.height > 0 and adj_close.height > 0 and dividend_history.height > 0):
                continue

            latest_close = float(view.latest_close()[symbol])
            close_series = closes["close"].to_list()
            adj_close_series = adj_close["adj_close"].to_list()
            dividend_series = [float(div) for div in dividend_history["dividend"].to_list()]

            if len(close_series) < self._window or len(adj_close_series) < self._window:
                continue

            earnings_volatility = (sum([(close - latest_close) ** 2 for close in close_series]) / self._window) ** 0.5
            dividend_yield = sum(dividend_series) / sum(adj_close_series)

            dividend_yield_score = dividend_yield_score.with_columns(
                pl.Series(symbol, [dividend_yield])
            ).with_columns(
                pl.Series(symbol + "_volatility", [earnings_volatility])
            )

        if dividend_yield_score.height == 0:
            return Signal(information_available_at=stamp, weights={})

        dividend_yield_score = dividend_yield_score.sort(
            (pl.col("dividend_yield") * 30.0 / 100) + ((1 - pl.col("volatility")) * 70.0), descending=True
        ).head(self._top_n)

        picks: list[str] = [symbol for symbol in dividend_yield_score.columns if not symbol.endswith("_volatility")]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest