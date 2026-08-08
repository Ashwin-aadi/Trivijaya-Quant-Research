from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy capitalizes on breakout continuation patterns in Indian equity markets by "
        "identifying strong breakouts with volume confirmation. It leverages the tendency for stocks that break out of a consolidation phase to continue their trend direction post-breakout."
    )

    def __init__(self, window: int = 60, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_close")
            )
            high = float(data[f"{symbol}_high"].max())
            low = float(data[f"{symbol}_low"].min())

            breakout_high = (data.filter(pl.col(f"{symbol}_close") > high)).shape[0] > 0
            breakout_low = (data.filter(pl.col(f"{symbol}_close") < low)).shape[0] > 0

            volume_threshold = float(history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_volume")
            ).filter(
                (pl.col(f"{symbol}_close").max() == history[f"{symbol}_adj_close"])
                & (pl.col(f"{symbol}_close") > high)
            ).shape[0] * 1.5)

            if breakout_high:
                entry_price = float(history.filter(
                    pl.col("session_date") == stamp
                )[f"{symbol}_adj_close"])
                take_profit = entry_price + (high - entry_price) * 0.02
                stop_loss = entry_price - (high - entry_price) * 0.01
                picks.append((symbol, "long", entry_price, take_profit, stop_loss))
            elif breakout_low:
                entry_price = float(history.filter(
                    pl.col("session_date") == stamp
                )[f"{symbol}_adj_close"])
                take_profit = entry_price - (entry_price - low) * 0.02
                stop_loss = entry_price + (entry_price - low) * 0.01
                picks.append((symbol, "short", entry_price, take_profit, stop_loss))

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        sorted_picks = sorted(picks, key=lambda x: abs(x[3] - x[4]), reverse=True)
        top_signals = sorted_picks[: self._top_n]
        weight = 1.0 / len(top_signals)
        signals_dict = {symbol: weight for _, symbol, *_ in top_signals}
        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in signals_dict.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest