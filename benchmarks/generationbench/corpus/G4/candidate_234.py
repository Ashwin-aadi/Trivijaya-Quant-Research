from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthIndexStrategy(Strategy):
    rationale = (
        "This strategy identifies and invests in stocks with higher relative strength indices (RSI) "
        "compared to the broader market index, such as Nifty 50. The RSI ranks each stock's performance "
        "relative to the market, and top-performing stocks are selected for inclusion in the portfolio."
    )

    def __init__(self, window: int = 14, top_percentage: float = 0.1) -> None:
        self._window = window
        self._top_percentage = top_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50 = list(view.symbols)
        symbols = [symbol for symbol in view.symbols if symbol not in nifty50]
        closes = view.closes(lookback=self._window)

        rsi_series: dict[str, float] = {}
        for symbol in symbols:
            adj_closes = closes[symbol].to_list()
            prices = pl.DataFrame({"close": adj_closes}).with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return")
            ).sort("close", descending=True).select(
                pl.col("return").rank(method="ordinal", descending=False)
            ).to_series(name=f"{symbol}_rsi")

            rsi_series[symbol] = float(prices[f"{symbol}_rsi"].mean())

        sorted_rsis = sorted(rsi_series.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in sorted_rsis[: int(len(symbols) * self._top_percentage)]]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest