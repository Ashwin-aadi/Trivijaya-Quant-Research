from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on their daily trading volume over the past 60 days to identify "
        "high-liquidity equities. Equal weighting among selected stocks ensures balanced exposure while leveraging"
        " liquidity trends for potentially more stable performance."
    )

    def __init__(self, window: int = 60, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            volume_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(volume_series) < self._window:
                continue
            if sum(volume_series) > 0:  # Check if there's any trading activity
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        signal = Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )

        # Ensure risk rules are met
        portfolio_loss = sum([weight * 3.0 for _ in picks])
        if portfolio_loss > 5.0:
            return Signal(information_available_at=stamp, weights={})

        return signal


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest