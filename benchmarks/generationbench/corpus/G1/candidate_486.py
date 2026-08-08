from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthStrategy(Strategy):
    rationale = (
        "This strategy selects stocks that have outperformed the NIFTY 100 index over a "
        "recent period. Outperforming stocks are expected to continue their positive trend."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty100_closes = [f"NIFTY100_{i}" for i in range(len(view.symbols))]
        other_closes = view.closes(lookback=self._window)
        if any(symbol not in other_closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty100_returns = _calculate_returns(history[nifty100_closes])
        other_returns = {symbol: _calculate_returns(other_closes[[symbol]])[symbol] for symbol in view.symbols}

        outperformers: list[str] = []
        for symbol, return_value in other_returns.items():
            if (return_value / nifty100_returns[symbol]) > 1.2:
                outperformers.append(symbol)

        outperformers = outperformers[: self._top_n]
        if not outperformers:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(outperformers)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in outperformers}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_returns(frame: pl.DataFrame) -> dict[str, float]:
    latest_close = frame.select(pl.last("adj_close")).to_numpy().flatten()[0]
    prev_close = (
        (frame.sort("session_date").select(pl.first("adj_close"))).to_numpy()
        .flatten()[1:]
    )
    returns = {symbol: (latest / prev - 1) for symbol, latest, prev in zip(frame.columns[1:], latest_close, prev_close)}
    return returns