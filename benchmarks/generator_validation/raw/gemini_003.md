## cross-sectional momentum
```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum capitalizes on the persistence of relative performance across a universe. "
        "By looking at the 12-month return and skipping the most recent month, we avoid short-term reversal effects. "
        "This assumes that the macroeconomic drivers benefiting the top decile will persist over the medium term."
    )

    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 10) -> None:
        self._lookback = lookback
        self._skip = skip
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)
        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            vals = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(vals) < self._lookback:
                continue
            ret = (vals[-self._skip - 1] / vals[0]) - 1.0
            returns[symbol] = ret

        sorted_syms = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)
        picks = sorted_syms[:self._top_n]

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
```

## short-horizon mean reversion
```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ShortHorizonMeanReversion(Strategy):
    rationale = (
        "Equities frequently overreact to short-term news, leading to temporary price dislocations. "
        "By identifying stocks that have suffered the steepest declines over a 5-day window, we capture the liquidity premium of providing counter-trend flow. "
        "This strategy assumes that extreme short-term moves are driven by transient supply-demand imbalances."
    )

    def __init__(self, window: int = 5, bot_n: int = 5) -> None:
        self._window = window
        self._bot_n = bot_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            vals = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(vals) < self._window:
                continue
            returns[symbol] = (vals[-1] / vals[0]) - 1.0

        sorted_syms = sorted(returns.keys(), key=lambda s: returns[s])
        picks = sorted_syms[:self._bot_n]

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
```

## volatility-scaled trend following
```python
from __future__ import annotations

import math
from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "Trend following captures persistent price trajectories, but unscaled momentum exposes the portfolio to volatility spikes. "
        "By scaling a medium-term trend signal inversely by its realized volatility, we risk-parity the trend bets. "
        "This strategy dynamically allocates more capital to stable, steady trends."
    )

    def __init__(self, window: int = 60, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            vals = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(vals) < self._window:
                continue
            
            rets = [(vals[i] / vals[i-1]) - 1.0 for i in range(1, len(vals))]
            mean_ret = sum(rets) / len(rets)
            variance = sum((r - mean_ret)**2 for r in rets) / (len(rets) - 1)
            ann_vol = math.sqrt(variance) * math.sqrt(252)
            
            if ann_vol < 1e-6:
                continue
                
            trend = (vals[-1] / vals[0]) - 1.0
            if trend > 0:
                scores[symbol] = trend / ann_vol

        sorted_syms = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
        picks = sorted_syms[:self._top_n]
        
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
```

## liquidity-screened equal weighting
```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeight(Strategy):
    rationale = (
        "Equally weighting an index universe removes capitalization biases, harvesting the size premium. "
        "However, true equal weighting can lead to excessive trading costs in illiquid names. "
        "By screening for the highest average trading volume, we build a liquid pseudo-equal-weight proxy."
    )

    def __init__(self, window: int = 20, top_n: int = 50) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols_vol: dict[str, list[float]] = {s: [] for s in view.symbols}
        
        sym_col = history["symbol"].to_list()
        vol_col = history["volume"].to_list()
        
        for s, v in zip(sym_col, vol_col):
            if s in symbols_vol and v is not None:
                symbols_vol[s].append(float(v))
                
        avg_vols: dict[str, float] = {}
        for s, vols in symbols_vol.items():
            if len(vols) >= self._window * 0.8:
                avg_vols[s] = sum(vols) / len(vols)
                
        sorted_syms = sorted(avg_vols.keys(), key=lambda k: avg_vols[k], reverse=True)
        picks = sorted_syms[:self._top_n]
        
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
```

## price-level reversion against a trailing reference
```python
from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReferenceReversion(Strategy):
    rationale = (
        "Asset prices often stretch too far from their moving averages before reverting to the mean. "
        "By identifying stocks trading significantly below their 20-day trailing reference, we isolate overextended names. "
        "This targets mean reversion in a longer context than standard daily noise."
    )

    def __init__(self, window: int = 20, bot_n: int = 5) -> None:
        self._window = window
        self._bot_n = bot_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        deviations: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            vals = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(vals) < self._window:
                continue
                
            ma = sum(vals) / len(vals)
            current = vals[-1]
            
            if ma > 0:
                dev = (current - ma) / ma
                deviations[symbol] = dev

        sorted_syms = sorted(deviations.keys(), key=lambda s: deviations[s])
        picks = sorted_syms[:self._bot_n]
        
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
```
