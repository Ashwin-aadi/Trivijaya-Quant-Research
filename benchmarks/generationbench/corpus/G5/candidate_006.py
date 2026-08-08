from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in stocks that have outperformed the broad market over a recent period "
        "can lead to higher returns. This strategy identifies such stocks by comparing "
        "individual stock performance against the average performance of the S&P 500 index."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_avg = (
            history.select(pl.col("adj_close").mean().alias("market_avg"))
            .group_by(pl.lit(True))
            .agg(pl.col("market_avg"))
            .collect()
            .get_column("market_avg")
            .to_list()[0]
        )
        if market_avg == 0:
            return Signal(information_available_at=stamp, weights={})

        symbol_performance = {}
        for symbol in view.symbols:
            symbol_data = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("adj_close"),
                    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
                )
                .sort("session_date")
            )

            if symbol_data.height < self._window:
                continue

            avg_return = (
                symbol_data.select(pl.col("return").mean().alias("avg_return"))
                .collect()
                .get_column("avg_return")
                .to_list()[0]
            )

            symbol_performance[symbol] = avg_return / market_avg

        sorted_symbols = [
            s for _, s in sorted(
                symbol_performance.items(), key=lambda item: item[1], reverse=True
            )
        ][:5]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest