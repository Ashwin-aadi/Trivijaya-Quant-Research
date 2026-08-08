from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MomentumLiquidityComposite(Strategy):
    rationale = (
        "This strategy exploits the interplay between sector-specific momentum and market-wide liquidity. "
        "It identifies sectors showing recent positive momentum and trades stocks within these sectors that are currently undervalued relative to historical norms."
    )

    def __init__(self, momentum_window: int = 30, liquidity_window: int = 5) -> None:
        self._momentum_window = momentum_window
        self._liquidity_window = liquidity_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._momentum_window + self._liquidity_window)
        if closes.height < self._momentum_window + self._liquidity_window:
            return Signal(information_available_at=stamp, weights={})

        sectors: dict[str, list[tuple[float, str]]] = {}
        for symbol in view.symbols:
            sector = symbol.split(".")[0]
            if sector not in sectors:
                sectors[sector] = []

            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._momentum_window + 1:  # +1 because of shift
                continue

            momentum_signal = (values[-1] - values[0]) / values[0]
            liquidity_signal = _vwap_change(symbol, closes)
            sectors[sector].append((momentum_signal, liquidity_signal, symbol))

        selected_sectors = sorted(sectors.items(), key=lambda x: max([t[0] for t in x[1]]), reverse=True)[:5]

        picks: list[str] = []
        for sector, items in selected_sectors:
            top_stock = max(items, key=lambda x: x[1])
            picks.append(top_stock[2])

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _vwap_change(symbol: str, closes: pl.DataFrame) -> float:
    latest_close = view.latest_close()[symbol]
    recent_vwaps = [float(v) for v in
                    (closes
                     .with_columns((pl.col("adj_close") * pl.col("volume")) / pl.col("volume").sum().alias("vwap"))
                     .sort("session_date", descending=True)
                     .select(pl.col("vwap"))
                     .head(self._liquidity_window)
                     ["vwap"]
                     .to_list())]

    recent_vwap = sum(recent_vwaps) / self._liquidity_window
    historical_vwap = _historical_vwap(symbol, closes)

    return (recent_vwap - latest_close) / historical_vwap


def _historical_vwap(symbol: str, closes: pl.DataFrame) -> float:
    recent_prices = [float(v) for v in
                     (closes
                      .sort("session_date", descending=True)
                      .select(pl.col("adj_close"))
                      .head(30)["adj_close"]
                      .to_list())]

    historical_vwap = sum(recent_prices) / 30.0

    return historical_vwap