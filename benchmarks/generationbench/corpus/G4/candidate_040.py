from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion50d(Strategy):
    rationale = (
        "This strategy exploits mean reversion in stock prices relative to their historical levels. "
        "Stocks often return to their long-term average price levels due to market inefficiencies or mean reversion tendencies. "
        "By identifying stocks trading below a trailing reference level (e.g., a 50-day moving average) as potential buys, "
        "and selling those above this level, the strategy aims to capitalize on temporary deviations from fair value."
    )

    def __init__(self, window: int = 50, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sma_col_name = f"sma_{self._window}"
        history = (
            history.with_columns(
                (pl.col("adj_close").rolling_mean(self._window)).alias(sma_col_name)
            )
            .sort("session_date", descending=False)
            .with_columns(
                ((pl.col("adj_close") - pl.col(sma_col_name)).abs()).alias("deviation")
            )
        )

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            history_symbol = history[symbol]
            sma_symbol = history[sma_col_name][symbol]

            latest_close = float(view.latest_close()[symbol])
            latest_sma = float(sma_symbol[-1])
            latest_deviation = abs(latest_close - latest_sma)

            if latest_deviation >= max(history_symbol[symbol].to_list()[-self._window :]):
                picks.append(symbol)

        picks = sorted(picks, key=lambda s: (latest_close - sma_symbol[s][-1]), reverse=True)[: self._top_n]
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