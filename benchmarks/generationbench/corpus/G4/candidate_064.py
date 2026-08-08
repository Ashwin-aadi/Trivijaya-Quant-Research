from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingReference(Strategy):
    rationale = (
        "Prices often revert to a historical average after deviating significantly due to market "
        "noise or speculative behavior. This strategy exploits this mechanism by identifying stocks "
        "that have traded above or below their trailing 52-week reference levels and are likely to "
        "revert back towards these levels."
    )

    def __init__(self, window: int = 52, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        highs = history.select(
            pl.col("symbol"), pl.col("high").over("symbol").last().alias("trailing_high")
        )
        lows = history.select(
            pl.col("symbol"), pl.col("low").over("symbol").first().alias("trailing_low")
        )

        closes = view.closes(lookback=self._window)
        deviations = closes.join(highs, on="symbol", how="left").join(lows, on="symbol", how="left")

        for symbol in view.symbols:
            if symbol not in deviations.columns or (deviations[symbol].is_null().sum() > 0):
                continue
            current_close = float(deviations[stamp]["close"])
            trailing_high = float(deviations[stamp]["trailing_high"])
            trailing_low = float(deviations[stamp]["trailing_low"])

            if current_close < trailing_low:
                deviation_score = (current_close - trailing_low) / trailing_low
            elif current_close > trailing_high:
                deviation_score = (current_close - trailing_high) / trailing_high
            else:
                continue

            deviations.update(
                pl.col("symbol").set_value(symbol, deviation_score),
                in_place=True,
            )

        top_symbols = (
            deviations.sort("close", descending=False)
            .select(["symbol"])
            .head(self._top_n)["symbol"]
            .to_list()
        )
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest