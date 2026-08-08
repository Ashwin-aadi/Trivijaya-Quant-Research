from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "This strategy aims to capitalize on mean-reverting behavior in stock prices by "
        "trading against price levels relative to a trailing reference. By systematically "
        "entering trades in the opposite direction of recent price action relative to this "
        "trailing reference, the strategy seeks to profit from temporary deviations from "
        "long-term trends."
    )

    def __init__(self, window: int = 200, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        tmas = history.group_by("symbol").agg(
            (pl.col("adj_close").shift(-1).rolling_mean(self._window)).alias(f"tma_{self._window}")
        )
        latest_closes = view.closes(lookback=None)
        if len(tmas) == 0 or len(latest_closes.columns) < self._top_n + 1:
            return Signal(information_available_at=stamp, weights={})

        merged = tmas.join(
            latest_closes,
            on="symbol",
            how="inner"
        ).sort("symbol")

        signals: list[str] = []
        for symbol in merged["symbol"]:
            if (
                float(merged[merged["symbol"] == symbol]["adj_close"].to_list()[-1]) -
                float(merged[merged["symbol"] == symbol][f"tma_{self._window}"].to_list()[-1])
            ) < -0.5:
                signals.append(symbol)

        if len(signals) < 1:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / min(self._top_n, len(signals))
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals[: self._top_n]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest