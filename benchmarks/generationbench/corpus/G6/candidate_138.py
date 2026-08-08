from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIRelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks based on their Relative Strength (RS) against the broader NIFTY 50 index. "
        "Stocks with an RSI below 30 and showing relative strength compared to the NIFTY 50 are entered, "
        "ensuring momentum and resilience in the portfolio."
    )

    def __init__(self, window_short: int = 20, window_long: int = 50, top_n: int = 10) -> None:
        self._window_short = window_short
        self._window_long = window_long
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_long + 1)
        if closes.height < self._window_long + 1:
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = view.closes(lookback=self._window_long + 1).select(pl.col("NIFTY_50").alias("nifty50_adj_close"))
        if nifty50_closes.height < self._window_long + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol == "NIFTY_50" or symbol not in closes.columns or symbol not in nifty50_closes.columns:
                continue

            stock_closes = closes[symbol].to_list()
            nifty50_closes_val = [float(v) for v in nifty50_closes["nifty50_adj_close"].drop_nulls().to_list()]

            if len(stock_closes) < self._window_short or len(nifty50_closes_val) < self._window_long:
                continue

            stock_rsi = _calculate_rsi(stock_closes, window=self._window_short)
            nifty50_rsi = _calculate_rsi(nifty50_closes_val, window=self._window_short)

            if stock_rsi[-1] < 30 and stock_rsi[-1] > nifty50_rsi[-1]:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

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


def _calculate_rsi(prices: list[float], window: int) -> pl.Series:
    delta = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gain = [x if x > 0 else 0 for x in delta]
    loss = [-x if x < 0 else 0 for x in delta]

    avg_gain = pl.Series(gain).rolling_mean(window=window)
    avg_loss = pl.Series(loss).abs().rolling_mean(window=window)

    rs = (avg_gain / avg_loss).drop_nulls()
    rsi = 100 - (100 / (1 + rs))
    return rsi