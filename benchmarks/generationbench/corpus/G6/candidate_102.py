from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks outperforming the NIFTY 50 index using relative strength. "
        "By focusing on RSI and volume changes, it aims to capture strong performers while managing risk."
    )

    def __init__(self, window: int = 14, top_n_high_risk: int = 5, top_n_low_risk: int = 15) -> None:
        self._window = window
        self._top_n_high_risk = top_n_high_risk
        self._top_n_low_risk = top_n_low_risk

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        nifty50_closes = view.closes(lookback=self._window).select(
            pl.col("session_date"), *[pl.col(s) for s in view.symbols if "NIFTY" in s]
        )
        symbols = history["symbol"].to_list()

        rsi = _calculate_rsi(history)
        volume_change = (history["volume"] / history["volume"].shift(1)).drop_nulls().alias("vol_change")
        nifty50_closes = nifty50_closes.join(rsi, on="session_date", how="inner").select(
            pl.col("symbol"), "adj_close", "rsi"
        )
        combined = history.join(nifty50_closes, on="session_date", how="left")

        filtered = combined.filter(
            (combined["rsi"] < 30) & (combined["vol_change"].gt(1))
        ).select("symbol").unique().to_list()
        if not filtered:
            return Signal(information_available_at=stamp, weights={})

        high_risk = [s for s in symbols if s in filtered][:self._top_n_high_risk]
        low_risk = [s for s in symbols if s in filtered and s not in high_risk][:self._top_n_low_risk]

        total_weight = 0.6 + 0.4 / (len(low_risk) + 1)
        weights = {s: total_weight * (0.6 / self._top_n_high_risk + 0.4 / (len(low_risk) + 1)) for s in high_risk}
        for i, s in enumerate(low_risk):
            weights[s] = total_weight * (0.4 / (len(low_risk) + 1))

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_list()[0]
    assert isinstance(newest, date)
    return newest


def _calculate_rsi(history: pl.DataFrame) -> pl.DataFrame:
    delta = history.select("close", "adj_close").select(
        (pl.col("adj_close") - pl.col("close")).alias("delta")
    )
    up = delta.filter(pl.col("delta") > 0).with_column(pl.col("delta").alias("up"))
    down = delta.filter(pl.col("delta") < 0).with_column((-1 * pl.col("delta")).alias("down"))

    up_sum = up.groupby("symbol").agg(
        (pl.col("up").sum()).alias("up_sum")
    )
    down_sum = down.groupby("symbol").agg(
        (pl.col("down").sum()).alias("down_sum")
    )

    rsi = up_sum.join(down_sum, on="symbol", how="inner").with_columns(
        (((100 - 100 / (1 + pl.col("up_sum") / pl.col("down_sum")))).alias("rsi"))
    )
    return rsi