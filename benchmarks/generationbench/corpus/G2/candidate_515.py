from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day momentum of a stock "
        "and its relative strength compared to the NIFTY 100 index. The idea is that stocks with "
        "both strong momentum and outperforming relative strength may exhibit higher returns."
    )

    def __init__(self, window_momentum: int = 20, threshold_rel_strength: float = 0.8) -> None:
        self._window_momentum = window_momentum
        self._threshold_rel_strength = threshold_rel_strength

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window_momentum + 1)

        if closes.height < self._window_momentum + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate momentum
        momentum_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window_momentum + 1:
                continue
            last_close = values[-1]
            mean_close = sum(values[:-1]) / (self._window_momentum - 1)
            momentum_score = (last_close - mean_close) / mean_close
            momentum_scores[symbol] = momentum_score

        # Calculate relative strength compared to NIFTY 100 index
        nifty_100_closes = closes.get_column(view.symbols[0]).to_list()
        if len(nifty_100_closes) < self._window_momentum + 1:
            return Signal(information_available_at=stamp, weights={})

        last_nifty_100_close = nifty_100_closes[-1]
        mean_nifty_100_close = sum(nifty_100_closes[:-1]) / (self._window_momentum - 1)
        rel_strength_score = (last_nifty_100_close - mean_nifty_100_close) / mean_nifty_100_close

        # Find symbols with both high momentum and relative strength
        picks: list[str] = []
        for symbol, momentum in momentum_scores.items():
            if momentum >= 0.2 and rel_strength_score > self._threshold_rel_strength:
                picks.append(symbol)

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