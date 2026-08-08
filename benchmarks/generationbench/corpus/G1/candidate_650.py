from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies trending stocks by combining volatility and recent price movement. "
        "High volatility combined with a strong upward trend is indicative of a potentially robust trend."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].unique().to_list()]

        signals: dict[str, float] = {}
        for symbol in symbols:
            df = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(
                    pl.col("close").alias("c"),
                    (pl.col("close").shift(-1) / pl.col("close").shift(1) - 1).alias("r")
                )
            )

            if df.height < self._vol_window + 2:
                continue

            price_return = df.select(pl.sum(pl.col("r"))).to_numpy()[0][0]
            volatility = (
                df.filter(pl.col("r").is_not_nan())
                .select(
                    (pl.col("r") - pl.col("r").mean()).pow(2)
                )
                .sort("c", descending=True)
                .head(self._vol_window)
                .select(pl.mean("r"))
                .to_numpy()[0][0]
            )

            if volatility == 0:
                continue

            trend_score = price_return / volatility
            signals[symbol] = trend_score

        sorted_signals = dict(
            sorted(signals.items(), key=lambda item: item[1], reverse=True)
        )
        top_symbols = list(sorted_signals.keys())[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight for s in top_symbols
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest