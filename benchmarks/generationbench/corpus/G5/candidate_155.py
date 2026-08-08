from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related signals: the 20-day closing price change "
        "and the relative strength index (RSI) over a shorter period. The idea is that both "
        "indicators can provide valuable insights when taken together."
    )

    def __init__(self, window_closing: int = 20, rsi_window: int = 7) -> None:
        self._window_closing = window_closing
        self._rsi_window = rsi_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_closing + self._rsi_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate closing price change
        changes = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("change")
        recent_changes = (
            history.with_columns(changes)
                   .sort("session_date", descending=False)
                   .select(pl.col("change").mean())
        )
        
        # Calculate RSI
        rsi_values = _calculate_rsi(history, self._rsi_window)

        if rsi_values.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_change = float(recent_changes["change"].to_list()[0])
        recent_rsi = float(rsi_values.select("rsi").rows()[-1]["rsi"])

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in rsi_values.columns or symbol not in history.select("adj_close").columns:
                continue
            rsi_value = rsi_values[symbol].to_list()[-1]
            change_value = history[symbol]["change"].mean().to_list()[0]

            # Consider a broader range of RSI values and closing price changes
            if (recent_rsi > 50 and mean_change > -0.02) or (rsi_value > 70 and change_value > 0):
                picks.append(symbol)

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


def _calculate_rsi(history: pl.DataFrame, window: int) -> pl.DataFrame:
    delta = history.select(pl.col("adj_close").diff().abs())
    up, down = delta.with_columns(
        (pl.col("value") * (pl.col("value") > 0)).alias("up"),
        (pl.col("value") * (pl.col("value") < 0)).alias("down")
    ).select(["up", "down"])

    ma_up = up.rolling_mean(window)
    ma_down = down.with_columns(pl.col("down").abs()).rolling_mean(window)

    rsi = (100 - 100 / (1 + ma_up / ma_down))
    return history.join(rsi, on="session_date", how="inner").select(["symbol", "rsi"])