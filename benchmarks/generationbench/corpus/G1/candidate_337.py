from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of asset prices to revert to their "
        "mean over a given period. Short-horizon mean reversion looks for extreme deviations "
        "from the average and bets on a return towards the mean."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_means: dict[str, float] = {}
        for symbol in view.symbols:
            mean_close = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    (pl.col("adj_close").mean().alias("avg"))
                )
                .collect()["avg"]
                .to_list()[0]
            )
            symbol_means[symbol] = mean_close

        closes = view.closes(lookback=self._window)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in symbol_means.keys():
                continue
            latest_close = float(closes[symbol].max().to_list()[0])
            mean_close = symbol_means[symbol]
            deviation = (latest_close - mean_close) / mean_close

            if abs(deviation) >= self._threshold:
                picks.append(symbol)

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