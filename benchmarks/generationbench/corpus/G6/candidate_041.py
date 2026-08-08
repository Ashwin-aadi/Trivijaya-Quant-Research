from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies stocks that have broken out of their trading range and are "
        "likely to continue moving in the direction of the breakout due to market sentiment and "
        "technical momentum. Entries are confirmed by a volume spike on the day of the breakout, "
        "ensuring strong buying or selling interest."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            history = view.history().select(
                "session_date", pl.col("adj_close").alias(f"{symbol}_close")
            )
            if history.height < self._window + 2:
                continue
            high, low = history[f"{symbol}_close"].tail(self._window).max(), history[
                f"{symbol}_close"
            ].tail(self._window).min()
            breakout_day = (
                history.filter(
                    (pl.col(f"{symbol}_close") > high) | (pl.col(f"{symbol}_close") < low)
                )
                .select("session_date")
                .head(1)
                .get(0, pl.Utf8)
                .item()
            )

            if breakout_day:
                breakout_day_history = history.filter(pl.col("session_date") == breakout_day).to_dict(
                    strip_colors=True
                )[f"{symbol}_close"][0]
                next_day = (history.select("session_date").filter(
                    pl.col("session_date") > breakouy_day
                ).head(1)
                .get(0, pl.Utf8)
                .item())
                next_day_history = history.filter(pl.col("session_date") == next_day).to_dict(strip_colors=True)[f"{symbol}_close"][0]

                if (next_day_history - breakout_day_history) > 0 and view.history().filter(
                    pl.col("adj_close").alias(f"{symbol}_close")
                ).select(pl.col(f"{symbol}_volume").head(2)).tail(1).to_dict(strip_colors=True)[f"{symbol}_volume"][0] > view.history().select(pl.col(f"{symbol}_volume")).mean().item():
                    breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[: self._top_n]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest