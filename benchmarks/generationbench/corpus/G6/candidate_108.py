from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class IntegratedMomentumStrategy(Strategy):
    rationale = (
        "This strategy focuses on identifying stocks with strong recent performance trends by combining elements from a conservative approach. It ranks stocks based on their average return over the past 60 days, selects the top 30%, and incorporates dynamic weekly re-evaluations to ensure adaptability to changing market conditions."
    )

    def __init__(self, lookback_period: int = 120, ranking_window: int = 60, top_percentile: float = 0.3, exit_lookback: int = 4, min_hold_days: int = 30) -> None:
        self._lookback_period = lookback_period
        self._ranking_window = ranking_window
        self._top_percentile = top_percentile
        self._exit_lookback = exit_lookback
        self._min_hold_days = min_hold_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.height < self._lookback_period:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=True)
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Rank symbols based on average return
        ranked = history.with_columns(
            pl.col("avg_return").rank(method="ordinal", descending=True).alias("rank")
        ).sort("rank")

        top_symbols = [row["symbol"] for _, row in ranked.iter_rows() if row["rank"] <= self._top_percentile * len(ranked)]
        
        # Check exit conditions
        exits = _check_exit_conditions(view, top_symbols)
        filtered_top_symbols = [s for s in top_symbols if s not in exits]

        # Generate Signal
        weights = {symbol: 1.0 / len(filtered_top_symbols) for symbol in filtered_top_symbols} if filtered_top_symbols else {}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _check_exit_conditions(view: MarketView, symbols: list[str]) -> set[str]:
    exits = set()
    for symbol in symbols:
        history = view.history(lookback=view.as_of - pl.Datetime("1M")).filter(pl.col("symbol") == symbol).sort("session_date", descending=True)
        if len(history) < 4 * 5:  # 4 weeks
            continue

        exit_day = _latest_visible(view)
        return_day = history.height - 30  # 30 trading days
        cum_return = (history.filter(pl.col("session_date") <= exit_day)["adj_close"].to_list()[-1] / history.filter(pl.col("session_date") <= history.sort("session_date", descending=True).row(return_day)[0])["adj_close"] - 1.0)

        if cum_return < -0.1:
            exits.add(symbol)
    return exits