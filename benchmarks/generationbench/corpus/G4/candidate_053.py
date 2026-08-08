from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuationStrategy(Strategy):
    rationale = (
        "This strategy identifies breakout continuation patterns in the Indian market. "
        "By confirming a significant price move beyond support or resistance levels and "
        "ensuring continued movement in that direction, it aims to capture profitable moves "
        "following breakouts."
    )

    def __init__(self, lookback: int = 30, breakout_threshold: float = 0.03, confirm_sessions: int = 5) -> None:
        self._lookback = lookback
        self._breakout_threshold = breakout_threshold
        self._confirm_sessions = confirm_sessions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_candidates: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.select(
                pl.col("session_date"), pl.col(symbol).alias(f"{symbol}_close")
            )
            prices = [float(v) for v in df[f"{symbol}_close"].drop_nulls().to_list()]
            if len(prices) < self._lookback:
                continue

            # Calculate the support and resistance levels
            high, low = max(prices), min(prices)
            recent_close = float(df.filter(pl.col("session_date") == stamp).select(f"{symbol}_close").item())
            support_level = recent_close * (1 - self._breakout_threshold)
            resistance_level = recent_close * (1 + self._breakout_threshold)

            # Check if there was a breakout
            for i in range(1, len(prices) - self._confirm_sessions):
                price = prices[i]
                if (
                    support_level < low and high < support_level
                    or resistance_level > high and low > resistance_level
                ):
                    if (price < support_level and recent_close >= support_level) \
                            or (price > resistance_level and recent_close <= resistance_level):
                        continue

                    # Check for continuation pattern over next confirm_sessions days
                    valid_continuation = True
                    for j in range(i + 1, i + self._confirm_sessions + 1):
                        if prices[j] < support_level and price >= support_level:
                            valid_continuation = False
                            break
                        elif prices[j] > resistance_level and price <= resistance_level:
                            valid_continuation = False
                            break
                    if not valid_continuation:
                        continue

                    # Volume-to-price ratio calculation
                    volume_ratio = df.filter(pl.col("session_date") >= history["session_date"][i])\
                                     .with_columns(
                                         (pl.col(symbol) / pl.col(symbol).shift(1) - 1.0).alias(f"{symbol}_return")
                                     )\
                                     .select((pl.col(symbol).sum() / f"{symbol}_return").alias("volume_ratio"))\
                                     .to_dict(as_pandas=False)[0]["volume_ratio"]
                    breakout_candidates[symbol] = volume_ratio * (price - support_level)

        if not breakout_candidates:
            return Signal(information_available_at=stamp, weights={})

        # Rank candidates
        sorted_candidates = sorted(breakout_candidates.items(), key=lambda x: (-x[1], -breakout_candidates[x[0]]))
        top_symbols = [symbol for symbol, _ in sorted_candidates[:20]]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest