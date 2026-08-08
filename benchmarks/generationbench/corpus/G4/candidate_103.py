from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DispersionCompressionStrategy(Strategy):
    rationale = (
        "This strategy exploits the theme of dispersion or range compression in Indian equity "
        "markets by dynamically adjusting sector exposures based on historical volatility. "
        "Sectors with higher-than-normal volatility are deemed dispersed, presenting potential "
        "arbitrage opportunities."
    )

    def __init__(self, window: int = 60, top_n_sectors: int = 8) -> None:
        self._window = window
        self._top_n_sectors = top_n_sectors

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sector_volatility = {}
        for symbol in view.symbols:
            sector = symbol.split(".")[0]  # Assume sectors are split by "."
            if sector not in sector_volatility:
                sector_volatility[sector] = []

            daily_returns = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    pl.col("session_date"),
                    (pl.col("adj_close") / pl.col("adj_open").shift(1) - 1.0).alias("return")
                )
                .with_columns((pl.col("return") * 100).round(2).alias("return_pct"))
            )

            if daily_returns.height < self._window:
                continue

            sector_volatility[sector].append(daily_returns.select("return_pct").to_numpy().ravel())

        weighted_sectors = []
        for sector, returns_list in sector_volatility.items():
            flat_returns = [item for sublist in returns_list for item in sublist]
            sector_mean_return = sum(flat_returns) / len(flat_returns)
            sector_std_dev = (sum((x - sector_mean_return) ** 2 for x in flat_returns) / len(returns_list)) ** 0.5
            weighted_sectors.append({"sector": sector, "std_dev": sector_std_dev})

        ranked_sectors = sorted(weighted_sectors, key=lambda x: x["std_dev"], reverse=True)
        top_sectors = [s["sector"] for s in ranked_sectors[: self._top_n_sectors]]

        weights = {}
        if top_sectors:
            weight_per_sector = 1.0 / len(top_sectors)
            for sector in top_sectors:
                symbols_in_sector = [
                    symbol for symbol in view.symbols if symbol.split(".")[0] == sector
                ]
                weights.update({symbol: weight_per_sector} for symbol in symbols_in_sector)

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest