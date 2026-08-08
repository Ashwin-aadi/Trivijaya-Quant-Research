from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression20d(Strategy):
    rationale = (
        "This strategy exploits sectors where stock prices exhibit significant volatility followed by range compression. By identifying and entering stocks during the range compression phase, we aim to profit from both initial dispersion and subsequent price normalization."
    )

    def __init__(self, window_volatility: int = 20, threshold_volatility: float = 1.5,
                 window_atr: int = 30, threshold_atr_decrease: float = 0.7, top_n: int = 10) -> None:
        self._window_volatility = window_volatility
        self._threshold_volatility = threshold_volatility
        self._window_atr = window_atr
        self._threshold_atr_decrease = threshold_atr_decrease
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_volatility + self._window_atr)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate volatility
        returns = (history["close"] / history["close"].shift(1) - 1).alias("return")
        vol_df = history.join(returns, on="session_date", how="left").drop_nulls()
        vol_df = vol_df.with_columns(
            pl.col("return").rolling_std(window=self._window_volatility, center=True).alias(f"volatility_{self._window_volatility}")
        )

        # Calculate ATR
        high_low_diff = (history["high"] - history["low"]).abs().alias("range")
        tr_high_low = ((history["high"] - history["close"].shift(1)).abs()).alias("tr_high_low")
        tr_high_close = ((history["high"] - history["adj_close"].shift(-1)).abs()).alias("tr_high_close")
        tr_low_close = ((history["low"] - history["adj_close"].shift(-1)).abs()).alias("tr_low_close")

        atr_df = (history.join(
            pl.concat_expr([high_low_diff, tr_high_low, tr_high_close, tr_low_close]),
            on="session_date", how="left").drop_nulls()
        ).with_columns(
            (pl.col(["range", "tr_high_low", "tr_high_close", "tr_low_close"])
             .max(axis=1).alias("true_range"))
        ).with_column(
            pl.col("true_range").rolling_mean(window=self._window_atr, center=True).alias(f"atr_{self._window_atr}")
        )

        # Join volatility and ATR
        combined = vol_df.join(atr_df, on="session_date", how="left")
        if combined.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Identify sectors with high recent volatility and decreasing ATR
        recent_volatility = combined[f"volatility_{self._window_volatility}"].shift(-10).to_list()
        recent_atr = combined[f"atr_{self._window_atr}"].shift(-10).to_list()
        filtered_symbols = []
        for symbol in view.symbols:
            if symbol not in combined.columns or (recent_volatility[-1] > self._threshold_volatility and
                                                  recent_atr[-1] < self._threshold_atr_decrease * recent_atr[0]):
                filtered_symbols.append(symbol)

        # Rank symbols based on breakout signals and volume
        ranking = {}
        for symbol in filtered_symbols:
            last_close = combined[symbol][-1]
            last_low = history[symbol]["low"][-10:].min()
            last_high = history[symbol]["high"][-10:].max()
            volume_ratio = float(combined[f"volume_{symbol}"][-1] / combined[f"volume_{symbol}"].rolling_mean(window=20).last())
            if last_close > last_high or last_close < last_low:
                ranking[symbol] = (last_high - last_low) + (volume_ratio * 10)

        # Select top N symbols
        picks = sorted(ranking, key=ranking.get, reverse=True)[:self._top_n]
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
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, pl.Datetime.date)
    return newest