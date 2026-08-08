from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class UnifiedStrategy(Strategy):
    rationale = (
        "This strategy leverages both earnings surprises (ES) to capture short-term market inefficiencies "
        "and volatility-based valuation (VBV) to mitigate risk through volatility screening. It aims to balance "
        "these elements to provide a robust approach."
    )

    def __init__(self, window_es: int = 20, top_n: int = 20, stop_loss_pct: float = -10.0, rebalance_period_days: int = 5 * 4) -> None:
        self._window_es = window_es
        self._top_n = top_n
        self._stop_loss_pct = stop_loss_pct
        self._rebalance_period_days = rebalance_period_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_es + 20)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate Earnings Surprises (ES)
        closes = history.select(pl.col("symbol"), pl.col("adj_close").shift(-1).alias("next_close"))
        es_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or "next_close" not in closes.columns:
                continue
            today_closes = [float(v) for v in history.select(symbol).to_numpy().flatten()]
            next_day_close = float(closes[symbol].filter(pl.col("symbol") == symbol).select("next_close").item())
            es_scores[symbol] = (today_closes[-1] - next_day_close) / today_closes[-1]

        # Calculate Volatility-Based Valuation (VBV)
        vvb_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            adj_closes = [float(v) for v in history.select(symbol).to_numpy().flatten()]
            vvb_scores[symbol] = pl.DataFrame({"adj_close": adj_closes}).select(
                (pl.col("adj_close").rolling_std(window_size=self._window_es, closed="both") / pl.col("adj_close").mean()).alias("volatility")
            ).item()

        # Combine ES and VBV scores
        combined_scores = {symbol: es_scores[symbol] + vvb_scores[symbol] for symbol in view.symbols if symbol in es_scores and symbol in vvb_scores}
        
        top_symbols = sorted(combined_scores.items(), key=lambda x: -x[1])[:self._top_n]
        
        # Check if any symbols meet the criteria
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(top_symbols)
        weights = {symbol: weight for symbol, _ in top_symbols}
        
        # Check stop-loss conditions (simplified for this example)
        latest_closes = view.closes(lookback=self._window_es).select(pl.col(view.symbols))
        if any([float(v) < current_price * (1 + self._stop_loss_pct / 100.0) for v, current_price in zip(latest_closes.to_numpy().flatten(), [view.latest_close()[s] for s in view.symbols])]):
            return Signal(information_available_at=stamp, weights={})
        
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest