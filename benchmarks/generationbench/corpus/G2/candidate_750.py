from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Assets that have outperformed the broader market over a given period are more likely "
        "to continue this trend due to mean reversion and selection bias. This strategy aims to "
        "identify such assets by comparing individual asset returns against the overall market return."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_return = float(history["adj_close"].mean().to_list()[-1] - history["adj_close"].mean().to_list()[0]) / history["adj_close"].mean().to_list()[0]

        relative_strength: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            asset_returns = [float(v) - market_return for v in (history.select(pl.col("symbol") == symbol)["adj_close"].to_list()[1:] + [0])]
            average_return = sum(asset_returns) / len(asset_returns)
            relative_strength[symbol] = average_return

        sorted_assets = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)
        top_n_assets = [symbol for symbol, _ in sorted_assets[:5]]
        if not top_n_assets:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_assets)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_assets}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest