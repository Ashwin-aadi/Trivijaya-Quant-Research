from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumEarningsFilter(Strategy):
    rationale = (
        "This strategy exploits strong short-term and medium-term momentum while considering "
        "earnings surprise to identify stocks with favorable performance metrics for investment."
    )

    def __init__(self, top_n: int = 25, earnings_threshold: float = 0.05) -> None:
        self._top_n = top_n
        self._earnings_threshold = earnings_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=60)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=10)

        def calculate_stm(symbol: str) -> float | None:
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < 10:
                return None
            high_10day = max(recent_closes)
            low_10day = min(recent_closes)
            today_close = history.filter(pl.col("session_date") == stamp).select(
                pl.col(symbol).alias("adj_close")
            ).item()
            if high_10day == 0 or low_10day == 0:
                return None
            return (today_close - high_10day) / (high_10day - low_10day)

        def calculate_mtm(symbol: str) -> float | None:
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(recent_closes) < 60:
                return None
            high_60day = max(recent_closes)
            low_60day = min(recent_closes)
            today_close = history.filter(pl.col("session_date") == stamp).select(
                pl.col(symbol).alias("adj_close")
            ).item()
            if high_60day == 0 or low_60day == 0:
                return None
            return (today_close - high_60day) / (high_60day - low_60day)

        def calculate_es(symbol: str) -> float | None:
            latest_close = view.latest_close()[symbol]
            earnings_data = history.filter(pl.col("session_date") == stamp).select(
                pl.col("adj_close").alias("earnings")
            ).item()
            if not (earliest := history["adj_close"].min()) or earliest == 0:
                return None
            return (latest_close - earnings_data) / earnings_data

        picks: list[str] = []
        for symbol in view.symbols:
            stm = calculate_stm(symbol)
            mtm = calculate_mtm(symbol)
            es = calculate_es(symbol)
            if stm is not None and mtm is not None and es is not None and stm > 0 and mtm > 0 and es > self._earnings_threshold:
                recent_close = closes[symbol].tail(1).to_list()[0][0]
                sma_10day = history.filter(pl.col("session_date") == stamp).select(
                    (pl.col(symbol) / pl.col(symbol).shift(9)).mean().alias("sma")
                ).item()
                if recent_close > sma_10day:
                    picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        stop_loss = {s: stamp - pl.duration(days=10) for s in picks}
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
            stop_losses=stop_loss
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest