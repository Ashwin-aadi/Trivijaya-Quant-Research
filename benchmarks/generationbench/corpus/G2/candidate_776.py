from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest that institutional traders are active in "
        "these stocks. Such moves can lead to sustained price trends and potentially "
        "profitable opportunities."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            hist = history[symbol].sort("session_date").to_pandas()
            open_prices = hist["open"].tolist()
            close_prices = hist["close"].tolist()
            volumes = hist["volume"].tolist()

            # Calculate daily returns
            rets = [(close / open) - 1 for open, close in zip(open_prices, close_prices)]
            if any(r < 0 for r in rets[-2:]):
                continue

            # Check for volume confirmation
            last_volume = volumes[-1]
            max_volume = max(volumes)
            if last_volume >= 1.5 * max_volume:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
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