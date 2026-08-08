from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedLowVolatility(Strategy):
    rationale = (
        "The strategy focuses on identifying stocks with lower historical volatility compared to peers while maintaining market exposure. This approach aligns with both designs' objectives by ensuring enhanced responsiveness and robust risk management."
    )

    def __init__(self, window: int = 20, top_n: int = 50, volume_threshold: float = 1e6) -> None:
        self._window = window
        self._top_n = top_n
        self._volume_threshold = volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 252)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        history = history.select(
            pl.col("symbol").alias("symbol"),
            (pl.col("high") - pl.col("low")).mean().over("symbol").alias("avg_range"),
            (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("log_ret"),
            (pl.col("volume").sum() / self._window).alias("avg_volume"),
        )
        
        if history.height < self._window * 252:
            return Signal(information_available_at=stamp, weights={})

        volatility_df = history.filter(pl.col("symbol").is_in(symbols))
        avg_log_ret = (volatility_df["log_ret"] / self._window).mean().alias("avg_log_ret")
        volatility = ((volatility_df["log_ret"] - avg_log_ret) ** 2).mean().alias("volatility")

        volatility_df = volatility_df.select(
            pl.col("symbol"),
            avg_log_ret,
            volatility,
            (pl.col("volume").sum() / self._window).alias("avg_volume"),
        )

        sorted_df = volatility_df.sort(["volatility", "avg_volume"], descending=[True, False])
        picks: list[str] = []
        for symbol in symbols:
            if (
                float(sorted_df.filter(pl.col("symbol") == symbol)["avg_volume"].first())
                < self._volume_threshold
            ):
                continue
            pick = sorted_df.filter(pl.col("symbol") == symbol).select(["volatility"]).head(1).row(0)[0]
            picks.append(symbol)
        
        top_n_picks = picks[: self._top_n]

        if not top_n_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest