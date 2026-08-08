from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityDispersionStrategy(Strategy):
    rationale = (
        "This strategy exploits dispersion in sectoral volatility during times of market "
        "dispersion and range compression. It identifies undervalued stocks within volatile "
        "sectors during dispersion periods and focuses on mean-reverting opportunities in "
        "range-compressed markets."
    )

    def __init__(self, window: int = 250, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )
        
        # Compute sectoral volatility using beta or standard deviation of returns
        sectors = set(history["symbol"].str.extract(r"^(.*)\.", expand=False).to_list())
        sector_beta_map = {}
        for sector in sectors:
            sector_history = history.filter(
                pl.col("symbol").str.contains(sector)
            ).group_by("session_date").agg(pl.col("return").std().alias("sector_return"))
            if not sector_history.is_empty():
                latest_beta = float(sector_history.sort("session_date", descending=True).head(1)["sector_return"])
                sector_beta_map[sector] = latest_beta

        # Identify dispersion and range compression periods
        dispersion_sectors = [s for s, v in sector_beta_map.items() if v > 0.5]
        compression_periods = history.groupby("session_date").agg(
            (pl.col("return").std().mean()).alias("market_volatility")
        ).sort("session_date", descending=True).head(1)["market_volatility"]
        
        # Determine market condition
        market_condition = "dispersion" if len(dispersion_sectors) > 0 else "compression"
        weight = 1.0 / self._top_n

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in sector_beta_map.keys():
                continue
            beta = sector_beta_map[symbol.split(".")[0]]
            recent_return = float(history.filter(pl.col("symbol") == symbol)["return"].tail(1).to_list()[0])
            
            if market_condition == "dispersion" and beta > 0.5:
                picks.append(symbol)
            elif market_condition == "compression":
                # Simple RSI-like check for range compression
                rsi = (recent_return < -0.02) or (recent_return > 0.02)
                if rsi:
                    picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest