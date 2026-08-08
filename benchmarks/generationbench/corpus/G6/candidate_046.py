from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalIntradayVolume(Strategy):
    rationale = (
        "Leveraging historical seasonality effects and intraday trading volume to identify "
        "opportunities in the Indian market. High trading volumes on previous days during "
        "seasonally active periods are indicative of strong investor interest, suggesting "
        "potential for higher returns."
    )

    def __init__(self, window: int = 30, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        high_volume_symbols = []
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)[
                ["session_date", "volume"]
            ]
            daily_volumes = [float(v) for v in symbol_history["volume"].to_list()]
            if len(daily_volumes) < self._window:
                continue
            recent_max_volume_idx = pl.col("volume").rank(method="dense", descending=True).head(1)
            recent_max_volume_date = (
                symbol_history.with_columns(recent_max_volume_idx.alias("max_volume_rank"))
                .filter(pl.col("max_volume_rank") == 1)
                .select("session_date")
                .to_list()[0][0]
            )
            if stamp - recent_max_volume_date < view.as_of - recent_max_volume_date:
                high_volume_symbols.append(symbol)

        high_volume_symbols = high_volume_symbols[: self._top_n]
        if not high_volume_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(high_volume_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in high_volume_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest