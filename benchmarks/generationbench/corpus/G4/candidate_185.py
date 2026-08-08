from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies and exploits price breakout patterns followed by strong volume confirmation. "
        "Breakouts are more likely to persist if they are supported by increased trading volumes, indicating broader consensus."
    )

    def __init__(self, window: int = 20, volume_window: int = 30, profit_target: float = 0.05, risk_limit: float = 0.02) -> None:
        self._window = window
        self._volume_window = volume_window
        self._profit_target = profit_target
        self._risk_limit = risk_limit

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        hh = closes.select(pl.col("adj_close").over(session_date).max().alias("hh"))
        ll = closes.select(pl.col("adj_close").over(session_date).min().alias("ll"))

        breakout_candles = (closes.join(hh, on="session_date", how="inner")
                            .join(ll, on="session_date", how="inner")
                            .with_columns(
                                (pl.col("adj_close") > pl.col("hh")).cast(pl.int32).alias("up_breakout"),
                                (pl.col("adj_close") < pl.col("ll")).cast(pl.int32).alias("down_breakout"),
                                (pl.col("volume").gt(pl.col("volume").mean().over(session_date) * 1.5)).alias("high_volume")
                            )
                           )

        breakout_candles = breakout_candles.filter((pl.col("up_breakout") == 1) & (pl.col("high_volume") == 1))
        if breakout_candles.height < 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = [symbol for symbol in view.symbols if symbol in breakout_candles.columns]
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