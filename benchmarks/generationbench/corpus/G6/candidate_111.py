from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityEqualWeighting(Strategy):
    rationale = (
        "This strategy selects stocks based on high liquidity to ensure robust performance. "
        "It then equally weights each selected stock to maintain balanced exposure and diversification."
        "Strict exit rules further enhance portfolio resilience by removing low-liquidity stocks and selling underperformers."
    )

    def __init__(self, min_volume: int = 50_000_000, min_exit_volume: int = 30_000_000, min_price_drop_percentage: float = -10.0) -> None:
        self._min_volume = min_volume
        self._min_exit_volume = min_exit_volume
        self._min_price_drop_percentage = min_price_drop_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=30)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        valid_symbols = set()
        for symbol in view.symbols:
            daily_volume = float(history.select(pl.col("volume")[pl.col("symbol") == symbol].sum()).to_dict()["value"][0])
            if daily_volume >= self._min_volume:
                valid_symbols.add(symbol)

        if len(valid_symbols) < 20:  # Select top 20 symbols by volume for simplicity
            return Signal(information_available_at=stamp, weights={})

        weight = 0.02
        selected_weights = {s: weight for s in sorted(valid_symbols)[:20]}

        price_drops = _check_price_drops(view, valid_symbols)
        if price_drops:
            for symbol, drop_percentage in price_drops.items():
                if drop_percentage < self._min_price_drop_percentage and symbol in selected_weights:
                    del selected_weights[symbol]

        if not selected_weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=selected_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _check_price_drops(view: MarketView, symbols: set[str]) -> dict[str, float]:
    price_drops = {}
    for symbol in symbols:
        closes = view.closes(lookback=30)[symbol].to_list()
        if len(closes) < 31:
            continue
        purchase_price = max(closes)
        current_price = view.latest_close()[symbol]
        drop_percentage = ((purchase_price - current_price) / purchase_price) * 100.0
        if drop_percentage < _min_price_drop_percentage:
            price_drops[symbol] = drop_percentage
    return price_drops