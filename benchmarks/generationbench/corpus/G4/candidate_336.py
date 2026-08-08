from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumStrategy(Strategy):
    rationale = (
        "This strategy capitalizes on the tendency for stocks with strong past performance (momentum) "
        "to continue outperforming in the near future. By identifying and weighting stocks based on their "
        "historical performance over a recent period, the portfolio aims to capture this momentum advantage."
    )

    def __init__(self, window: int = 120, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_dict()["symbol"]:
                continue
            prices = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("return")
                )
                .sort("session_date", descending=True)
                .head(self._window)
            )

            if prices.height < self._window:
                continue

            mean_return = float(prices.select(pl.col("return").mean()).to_dict()["return"][0])
            returns = [float(v) for v in prices["return"].drop_nulls().to_list()]
            momentum_score = sum([r - mean_return for r in returns]) / (self._window ** 0.5)

            if momentum_score > 0:
                picks[symbol] = abs(momentum_score)

        sorted_picks = {k: v for k, v in sorted(picks.items(), key=lambda item: item[1], reverse=True)}
        top_symbols = list(sorted_picks.keys())[: self._top_n]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest