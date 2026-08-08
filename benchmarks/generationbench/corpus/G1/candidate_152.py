from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment and "
        "are often followed by continuation of the trend. This strategy aims to identify such "
        "moves based on volume data."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes()
        latest_close_dates = set(latest_closes["session_date"].to_list())
        history_dates = {row[0] for row in history.select(["session_date"]).rows()}
        common_dates = sorted(history_dates.intersection(latest_close_dates))
        if len(common_dates) < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_prices = [float(view.latest_close()[symbol]) for symbol in view.symbols]
        volumes = history.select(["volume"] + list(view.symbols)).to_dict(as_series=False)
        price_changes = [(latest_prices[i] - latest_prices[i - 1]) / latest_prices[i - 1]
                         if i > 0 else 0.0 for i in range(len(latest_prices))]
        volume_changes = [float(volumes[symbol][-1] - volumes[symbol][-2])
                          if len(volumes[symbol]) >= 2 else 0.0 for symbol in view.symbols]

        directional_moves = {symbol: (price_changes[i], volume_changes[i])
                             for i, symbol in enumerate(view.symbols) if price_changes[i] > 0
                                                                 and volume_changes[i] > 0}

        if not directional_moves:
            return Signal(information_available_at=stamp, weights={})

        top_move = max(directional_moves.items(), key=lambda x: x[1][0])[0]
        weight = 1.0 / len(directional_moves)
        return Signal(
            information_available_at=stamp,
            weights={top_move: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest