from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSIBasedStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks that outperform or underperform the NIFTY 50 index "
        "using Relative Strength Index (RSI). It selects stocks with RSI below 30 and exits "
        "when their RSI rises above 70 relative to the NIFTY 50’s RSI, ensuring both "
        "conservative and ambitious criteria are met."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty50_history = history.filter(pl.col("symbol") == "NIFTY 50")
        other_stocks = [s for s in view.symbols if s != "NIFTY 50"]

        def calculate_rsi(df: pl.DataFrame) -> float:
            delta = df.select(
                (pl.col("close").shift(-1) - pl.col("close")).alias("delta"),
                (pl.col("high") - pl.col("low")).alias("range")
            )
            gain = delta.with_columns(pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0))
            loss = delta.with_columns(pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0))
            avg_gain = gain.select(
                (pl.col("delta").sum() / self._lookback_days).alias("avg_gain")
            )
            avg_loss = loss.select(
                (pl.col("delta").abs().sum() / self._lookback_days).alias("avg_loss")
            )
            rs = avg_gain.with_columns((pl.col("avg_gain") / pl.col("avg_loss")).alias("rs"))
            rsi = 100 - (100 / (1 + rs.select("rs").to_list()[0]))
            return float(rsi)

        nifty50_rsi = calculate_rsi(nifty50_history)
        other_stocks_signals: dict[str, tuple[float, float]] = {}
        for symbol in other_stocks:
            stock_history = history.filter(pl.col("symbol") == symbol)
            rsi = calculate_rsi(stock_history)
            other_stocks_signals[symbol] = (rsi, abs(rsi - nifty50_rsi))

        selected_stocks = [
            s for s, (rsi, diff) in sorted(other_stocks_signals.items(), key=lambda x: x[1][1])
            if rsi < 30
        ][:20]

        if not selected_stocks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_stocks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest