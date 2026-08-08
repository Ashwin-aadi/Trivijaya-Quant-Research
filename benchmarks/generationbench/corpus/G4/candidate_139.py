from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompression(Strategy):
    rationale = (
        "This strategy exploits the theme of dispersion or range compression in Indian equity markets by "
        "identifying stocks with significant price movements (dispersion) or narrow trading ranges (range "
        "compression). By aligning trades with these trends, the strategy aims to capitalize on expected future "
        "volatility shifts."
    )

    def __init__(self, window: int = 20, top_n: int = 5, stop_loss_percentage: float = 3.0) -> None:
        self._window = window
        self._top_n = top_n
        self._stop_loss_percentage = stop_loss_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        bollinger_bands = (
            history
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("sma"),
                (pl.col("adj_close") - pl.col("adj_close").mean()).std().over(pl.arange(0, self._window)).alias("stdev")
            )
        )

        bollinger_bands = (
            bollinger_bands
            .with_columns(
                ((pl.col("adj_close") - pl.col("sma")) / (2 * pl.col("stdev")).fill_null(1e-8)).alias("bb_ratio")
            )
            .sort("session_date", descending=False)
        )

        long_picks: list[str] = []
        short_picks: list[str] = []

        for symbol in view.symbols:
            if symbol not in bollinger_bands.columns:
                continue
            recent_values = [float(v) for v in bollinger_bands[symbol].to_list()[-self._window:]]
            bb_ratios = [v["bb_ratio"] for _, v in zip(recent_values, bollinger_bands[symbol].iter_rows())]

            if max(bb_ratios) >= 2.0:
                long_picks.append(symbol)
            elif min(bb_ratios) <= -2.0:
                short_picks.append(symbol)

        picks = long_picks + short_picks
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight_per_pick = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight_per_pick for s in long_picks
            },
            short_positions={
                s: weight_per_pick for s in short_picks
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest