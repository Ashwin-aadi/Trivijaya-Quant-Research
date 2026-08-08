from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DiwaliSeasonality(Strategy):
    rationale = (
        "Historically, the Indian stock market shows seasonality around key festivals and "
        "announcements. Specifically, stocks tend to outperform in the weeks leading up to and after Diwali, driven by increased liquidity and strategic financial decisions."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.height < self._window or len(closes.columns) != len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate average returns and volumes
        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue

            close_history = history.select([pl.col("session_date"), pl.col(symbol)])
            open_history = view.history(lookback=self._window).select(
                [pl.col("session_date"), f"open_{symbol}"]
            )
            high_history = view.history(lookback=self._window).select(
                [pl.col("session_date"), f"high_{symbol}"]
            )
            low_history = view.history(lookback=self._window).select(
                [pl.col("session_date"), f"low_{symbol}"]
            )
            volume_history = view.history(lookback=self._window).select(
                [pl.col("session_date"), f"volume_{symbol}"]
            )

            # Compute returns and volumes
            close_series = close_history[symbol].to_list()
            open_series = [float(v) for v in open_history[f"open_{symbol}"].drop_nulls().to_list()]
            high_series = [float(v) for v in high_history[f"high_{symbol}"].drop_nulls().to_list()]
            low_series = [float(v) for v in low_history[f"low_{symbol}"].drop_nulls().to_list()]
            volume_series = [int(v) for v in volume_history[f"volume_{symbol}"].drop_nulls().to_list()]

            # Calculate returns
            returns = [
                (close / open - 1.0) * (high - low) / high for close, open, high, low in zip(close_series[1:], open_series[:-1], high_series[1:], low_series[1:])
            ]

            avg_returns[symbol] = sum(returns) / len(returns)

        # Rank symbols based on average returns
        ranked_symbols = sorted(avg_returns.items(), key=lambda x: -x[1])
        picks = [symbol for symbol, _ in ranked_symbols[: self._top_n]]

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