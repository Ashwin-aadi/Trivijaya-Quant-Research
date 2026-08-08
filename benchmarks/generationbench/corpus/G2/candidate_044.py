from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are signals of strong buying or selling pressure. "
        "If a stock's volume and price move in the same direction significantly over several sessions, "
        "it suggests that market participants are aligning their trades with the prevailing trend."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)
        if history.is_empty() or history.height < self._window + 3:
            return Signal(information_available_at=stamp, weights={})

        volume_changes = []
        price_changes = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volumes = [
                float(v)
                for _, v in (
                    history[[symbol, "volume"]]
                    .sort("session_date")
                    .row_by_row()
                )
            ]
            if len(adj_closes) < self._window + 3:
                continue

            price_change = adj_closes[-1] - adj_closes[0]
            volume_change = volumes[-1] - volumes[0]

            if (price_change > 0 and volume_change > 0) or (
                price_change < 0 and volume_change < 0
            ):
                volume_changes.append(volume_change)
                price_changes.append(price_change)

        if not volume_changes or not price_changes:
            return Signal(information_available_at=stamp, weights={})

        avg_volume_change = sum(volume_changes) / len(volume_changes)
        avg_price_change = sum(price_changes) / len(price_changes)
        score = abs(avg_volume_change) / (avg_price_change + 1e-9)

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            volumes = [
                float(v)
                for _, v in (
                    history[[symbol, "volume"]]
                    .sort("session_date")
                    .row_by_row()
                )
            ]
            if len(adj_closes) < self._window + 3:
                continue

            price_change = adj_closes[-1] - adj_closes[0]
            volume_change = volumes[-1] - volumes[0]

            if (price_change > 0 and volume_change > 0) or (
                price_change < 0 and volume_change < 0
            ):
                score_symbol = abs(volume_change) / (abs(price_change) + 1e-9)
                if score_symbol >= score:
                    picks.append(symbol)

        top_n_symbols = picks[:5]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest