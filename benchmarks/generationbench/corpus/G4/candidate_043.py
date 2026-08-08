from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityDispersionStrategy(Strategy):
    rationale = (
        "This strategy identifies sectors with diverging price volatilities. By focusing on "
        "stocks within these sectors that exhibit increased or decreased volatility, the strategy "
        "harnesses periods of dispersion to generate profits."
    )

    def __init__(self, window: int = 20, top_bottom_n: int = 10) -> None:
        self._window = window
        self._top_bottom_n = top_bottom_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        sector_map = {symbol: "SECTOR" for symbol in symbols}  # Simplified sector mapping

        volatility_df = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")).abs().alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"),  # Rolling mean return
                 (pl.col("return") / pl.col("return").shift(1) - 1.0).alias("daily_volatility"))
            .drop_nulls()
        )

        volatility_df = (
            volatility_df.join(
                sector_map.into_frame(),
                on="symbol",
                how="inner"
            )
            .group_by("SECTOR")
            .agg(
                (pl.col("daily_volatility").std().alias("sector_std")),
                pl.col("daily_volatility").mean().alias("sector_avg"),
                (pl.col("daily_volatility") - pl.col("sector_avg")).abs().alias("dispersion_rank")
            )
        )

        high_dispersion_sectors = volatility_df.filter(
            (pl.col("sector_std") > 0.1) & (pl.col("dispersion_rank").rank(method="dense", descending=True).lt(self._top_bottom_n))
        ).select("SECTOR")

        signals: dict[str, float] = {}
        for sector in high_dispersion_sectors["SECTOR"]:
            sector_history = history.filter(pl.col("symbol").is_in(sector_map.keys())).group_by("symbol")
            top_volatility_stocks = (
                sector_history.agg(
                    (pl.col("daily_volatility") - pl.col("sector_avg")).abs().rank(method="dense", descending=True).alias("volatility_rank"),
                    pl.col("adj_close").mean().alias("avg_price")
                )
                .sort("volatility_rank")
                .select(["symbol", "avg_price"])
                .to_series()
            )[:self._top_bottom_n]

            bottom_volatility_stocks = (
                sector_history.agg(
                    (pl.col("daily_volatility") - pl.col("sector_avg")).abs().rank(method="dense").alias("volatility_rank"),
                    pl.col("adj_close").mean().alias("avg_price")
                )
                .sort("volatility_rank")
                .select(["symbol", "avg_price"])
                .to_series()
            )[:self._top_bottom_n]

            for symbol in top_volatility_stocks:
                signals[symbol] = 0.1
            for symbol in bottom_volatility_stocks:
                signals[symbol] = -0.1

        return Signal(information_available_at=stamp, weights={k: v for k, v in signals.items() if v != 0.0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest