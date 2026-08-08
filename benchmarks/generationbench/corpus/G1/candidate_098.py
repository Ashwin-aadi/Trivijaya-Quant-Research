from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling trades based "
        "on historical volatility. High volatility periods are expected to be more profitable."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility_history = history.group_by("symbol").agg(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).sort("session_date").select(
            pl.col("symbol"),
            (pl.col("returns").rolling_std(self._volatility_window)).alias("volatility"),
            ("close").last().alias("latest_close")
        )

        volatility_history = volatility_history.sort("volatility", descending=True).to_pandas()

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in volatility_history["symbol"].tolist():
                continue
            vol = volatility_history[volatility_history["symbol"] == symbol]["volatility"].values[0]
            latest_close = volatility_history[volatility_history["symbol"] == symbol]["latest_close"].values[0]

            if (vol > 0.1) and (latest_close >= history.filter(pl.col("symbol") == symbol)["adj_close"].sort().tail(3).last()):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest