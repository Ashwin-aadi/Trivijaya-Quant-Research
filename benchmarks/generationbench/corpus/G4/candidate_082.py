from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "The strategy exploits predictable seasonal patterns in the Indian market, "
        "capitalizing on historically observed trends during festivals and post-monsoon periods."
    )

    def __init__(self, window: int = 5 * 252, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sector_map = {
            "RETAIL": ["IT", "CONSUMER DISCRETIONARY"],
            "HOSPITALITY": ["HOSTELRY", "TRAVEL"],
            "AGRICULTURE": ["AGRIBUSINESS", "SEEDS"]
        }

        signals: dict[str, float] = {}
        for sector in sector_map:
            picks: list[tuple[float, str]] = []
            for symbol in view.symbols:
                if any(keyword in symbol for keyword in sector_map[sector]):
                    values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
                    if len(values) < self._window:
                        continue
                    recent_close = max(values[-10:], default=0.0)
                    mean_return = (values[-1] - recent_close) / recent_close
                    volume_mean = pl.col(symbol).mean().item()
                    score = 0.6 * mean_return + 0.4 * (volume_mean / volume_mean.mean())
                    picks.append((-score, symbol))
            top_stocks = sorted(picks)[:self._top_n]
            for _, stock in top_stocks:
                signals[stock] = 1.0 / len(top_stocks)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        return Signal(
            information_available_at=stamp, weights=signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest