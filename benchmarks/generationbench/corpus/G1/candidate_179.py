from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeCharStrategy(Strategy):
    rationale = (
        "This strategy combines the 20-day high close with a simple moving average "
        "crossover to identify potential buy signals. High closes indicate recent strength, "
        "while the moving average crossover suggests momentum and mean reversion."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200) -> None:
        self._short_window = short_window
        self._long_window = long_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window, self._long_window))
        if history.is_empty() or history.height < max(self._short_window, self._long_window):
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(pl.col("adj_close"))
        short_ma = (closes.rolling_mean(window_size=self._short_window) / 100).alias("short_ma")
        long_ma = (closes.rolling_mean(window_size=self._long_window) / 100).alias("long_ma")

        crossover_df = history.join(short_ma, on="session_date").join(long_ma, on="session_date")

        if not _find_buy_signals(crossover_df):
            return Signal(information_available_at=stamp, weights={})

        buy_symbols: list[str] = []
        for symbol in view.symbols:
            if crossover_df[symbol].max() > 1.0 and _is_high_close(symbol, history):
                buy_symbols.append(symbol)

        weight = 1.0 / len(buy_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in buy_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_buy_signals(df: pl.DataFrame) -> bool:
    df = df.filter(pl.col("long_ma") < 1.0).filter(pl.col("short_ma") > 1.0)
    return not df.is_empty()


def _is_high_close(symbol: str, history: pl.DataFrame) -> bool:
    symbol_df = history.select(pl.col(symbol))
    latest_close = float(symbol_df.max().to_list()[0])
    recent_closes = [float(v) for v in symbol_df.drop_nulls().sort("session_date", descending=True).head(20).to_list()]
    return all(latest_close > rc for rc in recent_closes)