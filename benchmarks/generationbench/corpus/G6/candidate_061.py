from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Stock prices tend to revert to their historical average levels over a specific period. "
        "This strategy aims to capture short-term price reversions while managing risks through "
        "strict criteria and a diversified portfolio structure."
    )

    def __init__(self, window: int = 20, threshold: float = 2.0, atr_window: int = 14) -> None:
        self._window = window
        self._threshold = threshold
        self._atr_window = atr_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._atr_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_df = (
            history
            .select(["symbol", "session_date", "close"])
            .with_columns(
                (pl.col("close") - pl.col("close").mean().over("symbol")).alias("diff"),
                ((pl.col("close") / pl.col("close").shift(1) - 1.0).abs()).alias("ret")
            )
        )

        atr_df = (
            history
            .select(["symbol", "session_date", "high", "low"])
            .with_columns(
                (pl.col("high").shift(-1) - pl.col("low")).abs().mean().over("symbol").alias("atr")
            )
        )

        combined = mean_reversion_df.join(atr_df, on=["symbol", "session_date"], how="inner")

        buy_symbols = []
        sell_symbols = []

        for symbol in view.symbols:
            if symbol not in combined.columns:
                continue
            data = combined.filter(pl.col("symbol") == symbol).to_numpy()
            session_dates = [d[1] for d in data]
            diffs = [float(d[2]) for d in data]
            atrs = [float(d[3]) for d in data]

            if len(session_dates) < self._window or len(atrs) < self._atr_window:
                continue

            latest_close_idx = session_dates.index(max(session_dates))
            latest_diff = diffs[latest_close_idx]
            latest_atr = atrs[latest_close_idx]

            if latest_diff <= -self._threshold and latest_atr > 0.5 * pl.col("close").mean().over("symbol"):
                buy_symbols.append(symbol)
            elif abs(latest_diff) >= self._threshold or (pl.col("ret") - pl.col("ret").rolling_mean(window=self._window)).max().abs() <= latest_atr:
                sell_symbols.append(symbol)

        buy_weights = {s: 1.0 / len(buy_symbols) for s in buy_symbols}
        sell_weights = {s: 1.0 / len(sell_symbols) for s in sell_symbols}

        weights = dict(**buy_weights, **sell_weights)
        if not weights:
            return Signal(information_available_at=stamp, weights={})
        
        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest