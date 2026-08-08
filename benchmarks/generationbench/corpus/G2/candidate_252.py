from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for recent winners to continue outperforming "
        "recent losers. This strategy bets on symbols that have shown strong positive returns in recent history."
    )

    def __init__(self, lookback_win: int = 20, lookback_loss: int = 20, top_n_win: int = 5) -> None:
        self._lookback_win = lookback_win
        self._lookback_loss = lookback_loss
        self._top_n_win = top_n_win

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_win + self._lookback_loss)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        winners = []
        losers = []

        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            close_history = [float(v) for v in history[symbol].to_list()]
            if len(close_history) < self._lookback_win + self._lookback_loss:
                continue

            returns = [(close_history[i] / close_history[i - 1] - 1.0) for i in range(1, len(close_history))]
            recent_returns = returns[-self._lookback_win:]
            if all(r > 0 for r in recent_returns):
                winners.append(symbol)
            elif all(r < 0 for r in recent_returns):
                losers.append(symbol)

        winners = sorted(winners, key=lambda s: sum(returns[i] for i in range(self._lookback_win) if s == close_history[i + self._lookback_loss]), reverse=True)[:self._top_n_win]
        losers = sorted(losers, key=lambda s: min(returns[i] for i in range(self._lookback_loss, self._lookback_win + self._lookback_loss) if s == close_history[i]), reverse=False)[:self._top_n_win]

        weight_win = 1.0 / len(winners)
        weight_loss = -1.0 / len(losers)

        weights = {s: weight_win for s in winners}
        for l in losers:
            weights[l] = weight_loss

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest