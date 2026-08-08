from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrengthIndexStrategy(Strategy):
    rationale = (
        "This strategy exploits relative strength by identifying stocks outperforming the "
        "S&P BSE Sensex. Strong stocks are selected based on their RSI above a threshold, ensuring "
        "a momentum-driven portfolio that adapts daily to recent market dynamics."
    )

    def __init__(self, window: int = 14, threshold: float = 70.0, top_n_percentage: float = 20.0) -> None:
        self._window = window
        self._threshold = threshold
        self._top_n_percentage = top_n_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if stamp is None:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        sensex_close = view.latest_close().get("BSE Sensex")
        if not sensex_close:
            return Signal(information_available_at=stamp, weights={})

        rsi_values: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol != "BSE Sensex":
                session_dates = closes["session_date"].to_list()
                adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
                sensex_adj_closes = [
                    float(v) for v in closes.filter(pl.col("symbol") == "BSE Sensex").select(["adj_close"]).to_dict()[0]["adj_close"]
                ]
                if len(adj_closes) < self._window:
                    continue

                price_changes = [(sensex_adj_closes[i] - sensex_adj_closes[i-1]) / sensex_adj_closes[i-1] for i in range(1, len(session_dates))]
                rsi = _compute_rsi(price_changes)
                rsi_values[symbol] = rsi

        ranked_symbols = sorted(rsi_values.items(), key=lambda x: x[1], reverse=True)
        top_n_count = int(len(view.symbols) * (self._top_n_percentage / 100))
        selected_symbols = [symbol for symbol, _ in ranked_symbols[:top_n_count]]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _compute_rsi(changes: list[float], window: int = 14) -> float:
    gains = [change if change > 0 else 0.0 for change in changes]
    losses = [-change if change < 0 else 0.0 for change in changes]

    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window

    rs = avg_gain / avg_loss if avg_loss != 0 else 0.0
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest