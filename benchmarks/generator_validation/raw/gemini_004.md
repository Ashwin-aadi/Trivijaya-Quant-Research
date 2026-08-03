## Cross-Sectional Momentum

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentumStrategy(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the empirical persistence of relative performance "
        "across equities over medium-term horizons. Stocks that have outperformed their peers "
        "over the past 3 to 12 months tend to continue outperforming due to investor underreaction "
        "and institutional capital flows. By filtering out the most recent month to avoid "
        "short-term reversal effects, the strategy captures sustained medium-term trend continuation. "
        "Equal weighting the top quintile systematically captures this factor premium across the universe."
    )

    def __init__(
        self, lookback: int = 126, skip_recent: int = 21, top_n: int = 5
    ) -> None:
        self._lookback = lookback
        self._skip_recent = skip_recent
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        scores: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback:
                continue
            p_start = values[-self._lookback]
            p_end = values[-self._skip_recent]
            if p_start > 0:
                ret = (p_end - p_start) / p_start
                scores.append((symbol, ret))

        if not scores:
            return Signal(information_available_at=stamp, weights={})

        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in scores[: self._top_n]]
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
```

## Short-Horizon Mean Reversion

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ShortHorizonMeanReversionStrategy(Strategy):
    rationale = (
        "Short-horizon mean reversion targets short-term supply and demand imbalances "
        "created by temporary market overreactions or liquidity shocks. Equities that experience "
        "sharp multi-day price declines often exhibit predictable short-term bounce-backs as "
        "buying interest returns at discounted valuations. By identifying cross-sectional extreme "
        "laggards over a 3-day window, the strategy captures short-term market efficiency restoration. "
        "Risk is managed by systematically diversification across high-conviction oversold constituents."
    )

    def __init__(
        self, window: int = 4, top_n: int = 5, threshold: float = -0.025
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        candidates: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            p_recent = values[-1]
            p_past = values[-self._window]
            if p_past > 0:
                ret = (p_recent - p_past) / p_past
                if ret < self._threshold:
                    candidates.append((symbol, ret))

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        candidates.sort(key=lambda x: x[1])
        picks = [s for s, _ in candidates[: self._top_n]]
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
```

## Volatility-Scaled Trend Following

```python
from __future__ import annotations

from datetime import date
import math

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendStrategy(Strategy):
    rationale = (
        "Volatility-scaled trend following aligns exposure with structural momentum while adjusting "
        "position size dynamically based on asset volatility. Stocks trading above their 50-day "
        "moving average are confirmed to be in sustained uptrends, mitigating false breakout risks. "
        "Scaling individual position weights inversely to 20-day realized volatility ensures equal "
        "risk contribution across universe constituents. This prevents volatile assets from dominating "
        "portfolio variance during turbulent market regimes."
    )

    def __init__(
        self, trend_window: int = 50, vol_window: int = 20, top_n: int = 5
    ) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_window)
        if closes.height < self._trend_window:
            return Signal(information_available_at=stamp, weights={})

        eligible: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._trend_window:
                continue

            sma_50 = sum(values[-self._trend_window :]) / self._trend_window
            latest_price = values[-1]

            if latest_price > sma_50:
                vol_prices = values[-self._vol_window :]
                vol = _compute_volatility(vol_prices)
                if vol > 1e-6:
                    eligible.append((symbol, vol))

        if not eligible:
            return Signal(information_available_at=stamp, weights={})

        eligible.sort(key=lambda x: x[1])
        selected = eligible[: self._top_n]

        inv_vols = [1.0 / vol for _, vol in selected]
        sum_inv_vols = sum(inv_vols)

        weights = {
            sym: inv_vol / sum_inv_vols
            for (sym, _), inv_vol in zip(selected, inv_vols)
        }
        return Signal(information_available_at=stamp, weights=weights)


def _compute_volatility(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(variance)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```

## Liquidity-Screened Equal Weighting

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeightStrategy(Strategy):
    rationale = (
        "Liquidity-screened equal weighting mitigates capacity constraints and execution slippage "
        "by restricting the investment universe to top-tier liquid securities. Filtering for equities "
        "with the highest 20-day average daily rupee turnover ensures deep market depth and minimal "
        "market impact. Allocating equal capital across this liquid subset captures broad market exposure "
        "without market-cap concentration risk. Systematic rebalancing harvests rebalancing yield "
        "from short-term price fluctuations."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        hist = view.history(lookback=self._window)
        if hist.is_empty():
            return Signal(information_available_at=stamp, weights={})

        turnover_map: dict[str, list[float]] = {}
        for row in hist.iter_rows(named=True):
            sym = row.get("symbol")
            vol = row.get("volume")
            px = row.get("adj_close")
            if px is None:
                px = row.get("close")
            if (
                isinstance(sym, str)
                and sym in view.symbols
                and vol is not None
                and px is not None
            ):
                turnover_map.setdefault(sym, []).append(float(vol) * float(px))

        avg_turnover: list[tuple[str, float]] = []
        for sym, vals in turnover_map.items():
            if len(vals) >= self._window:
                avg_turnover.append((sym, sum(vals) / len(vals)))

        if not avg_turnover:
            return Signal(information_available_at=stamp, weights={})

        avg_turnover.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in avg_turnover[: self._top_n]]
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
```

## Price-Level Reversion against a Trailing Reference

```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReferenceReversionStrategy(Strategy):
    rationale = (
        "Price-level reversion against a trailing reference posits that equity prices fluctuate "
        "in mean-reverting cycles around a dynamic fundamental baseline. A 50-day exponential moving "
        "average establishes a smooth reference anchor representing intermediate equilibrium valuation. "
        "When an asset's spot price pulls back significantly below this baseline without fundamental "
        "breakdown, it creates an attractive mean-reversion entry point. Selecting the most discounted "
        "constituents systematically captures value recovery premiums."
    )

    def __init__(
        self, window: int = 50, top_n: int = 5, min_pullback: float = -0.03
    ) -> None:
        self._window = window
        self._top_n = top_n
        self._min_pullback = min_pullback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        deviations: list[tuple[str, float]] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            ema = _compute_ema(values[-self._window :])
            latest_price = values[-1]
            if ema > 0:
                dev = (latest_price - ema) / ema
                if dev <= self._min_pullback:
                    deviations.append((symbol, dev))

        if not deviations:
            return Signal(information_available_at=stamp, weights={})

        deviations.sort(key=lambda x: x[1])
        picks = [s for s, _ in deviations[: self._top_n]]
        weight = 1.0 / len(picks)

        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _compute_ema(prices: list[float]) -> float:
    if not prices:
        return 0.0
    k = 2.0 / (len(prices) + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = (p - ema) * k + ema
    return ema


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest
```
