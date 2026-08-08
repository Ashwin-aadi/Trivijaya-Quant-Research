from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PharmaceuticalRealEstateStrategy(Strategy):
    rationale = (
        "This strategy targets the pharmaceutical and real estate sectors in India by "
        "exploiting their weakly correlated behaviors. Pharmaceuticals benefit from steady"
        " growth, while real estate experiences market-driven volatility. By analyzing "
        "historical data, we can identify periods where both sectors move independently."
    )

    def __init__(self, pharma_window: int = 90, real_estate_window: int = 60) -> None:
        self._pharma_window = pharma_window
        self._real_estate_window = real_estate_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._pharma_window + self._real_estate_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        pharma_symbols = tuple(symbol for symbol in view.symbols if "PHARM" in symbol)
        real_estate_symbols = tuple(
            symbol for symbol in view.symbols if "REAL_ESTATE" in symbol
        )

        if not pharma_symbols or not real_estate_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Pharmaceutical Sector Analysis
        pharma_closes = history.filter(pl.col("symbol").is_in(pharma_symbols))[
            ["session_date", "symbol", "adj_close"]
        ]
        pharma_returns = (pharma_closes.sort(by="session_date")["adj_close"] /
                          pharma_closes.shift(1)["adj_close"] - 1.0).alias("return")
        pharma_moving_avg = (
            pharma_closes.with_columns(pharma_returns)
                         .group_by("symbol")
                         .agg(pl.col("return").mean().alias("avg_return"))
                         .sort(by="avg_return", descending=True)["avg_return"]
        ).to_list()

        # Real Estate Sector Analysis
        real_estate_closes = history.filter(pl.col("symbol").is_in(real_estate_symbols))[
            ["session_date", "symbol", "adj_close"]
        ]
        macro_indicator = _fetch_macro_indicator(history)  # Assume this function is defined elsewhere
        real_estate_corr = (
            real_estate_closes.join(macro_indicator, on="session_date")
                             .group_by("symbol")
                             .agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                                  pl.col("macro_indicator").mean().alias("indicator"))
                             .select(pl.col("symbol"), "return", "indicator")
        )
        real_estate_corr = (
            real_estate_corr.with_columns(
                (pl.col("return") * -1) / pl.col("indicator").rank(method="dense", descending=True).cast(pl.Float64)
            ).sort(by="return", descending=True)["symbol"]
        ).to_list()

        # Combine and Filter
        pharma_filtered = [s for s in pharma_symbols if pharma_moving_avg[s] > 0.1]
        real_estate_filtered = [s for s in real_estate_symbols if real_estate_corr.index(s) < len(real_estate_corr) / 4]

        picks = pharma_filtered + real_estate_filtered
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest

def _fetch_macro_indicator(history: pl.DataFrame) -> pl.DataFrame:
    # Dummy implementation; assume this fetches and returns a DataFrame with macro indicators
    return history.select(pl.col("session_date"), "macro_indicator")