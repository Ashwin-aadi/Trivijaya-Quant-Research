from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion strategies capitalize on the tendency of stock prices to revert "
        "to their historical average levels over a short period. High volatility and recent price "
        "deviations from moving averages indicate potential reversions."
    )

    def __init__(self, window: int = 50, lookback_volatility: int = 10, top_n: int = 20) -> None:
        self._window_sma = window
        self._lookback_volatility = lookback_volatility
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_sma + self._lookback_volatility)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        history = history.select(["session_date", "adj_close"])[symbols]

        # Calculate 50-day SMA
        sma_50 = (
            history.with_columns(
                (pl.col("adj_close").shift(-self._window_sma).rolling_mean(self._window_sma)).alias(f"sma_{self._window_sma}")
            )
            .sort("session_date")
            .select(pl.all().except_("session_date"))
            .to_dict(as_series=False)
        )

        # Calculate daily returns and volatility
        returns = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
        ).sort("session_date")

        volatility = (
            returns.select(pl.all().exclude("session_date"))
            .select((pl.col("returns").std(ddof=0)).alias(f"volatility_{self._lookback_volatility}"))
            .to_dict(as_series=False)
        )

        # Identify stocks that recently crossed below 50-day SMA after a short-term uptrend
        signals = []
        for symbol in symbols:
            if sma_50[symbol][-1] < volatility[symbol][-1]:
                sma_above_last_3_days = all(
                    sma_50[symbol][i] > sma_50[symbol][i + 1] for i in range(len(sma_50[symbol]) - 4, len(sma_50[symbol]) - 2)
                )
                if sma_above_last_3_days:
                    signals.append(symbol)

        # Rank and select top_n candidates
        picks = sorted(signals, key=lambda s: (sma_50[s][-1] - volatility[s][-1]), reverse=True)[: self._top_n]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest