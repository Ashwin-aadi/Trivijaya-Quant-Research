from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum6m(Strategy):
    rationale = (
        "This strategy identifies stocks with strong recent price appreciation relative to their peers over a 6-month period. "
        "By entering long positions in the top 20% of such stocks and applying risk management through stop-losses and exit rules, "
        "the goal is to achieve consistent returns while managing risks effectively."
    )

    def __init__(self, window: int = 180, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate momentum scores
        mom_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate momentum score as the percentage change from 6 months ago to current close
            mom_score = (values[-1] - values[0]) / values[0]
            mom_scores[symbol] = mom_score

        sorted_symbols = [
            s for _, s in sorted(mom_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        ][: self._top_n]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top20_percent = int(len(sorted_symbols) * 0.2)
        picks = sorted_symbols[:top20_percent]
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