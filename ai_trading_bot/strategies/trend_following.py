from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..indicators.ema_rsi_volume import compute_trend_indicators
from .base import Strategy, StrategyResult


@dataclass
class TrendFollowingStrategy(Strategy):
    name: str = "trend_following"
    volume_threshold: float = 1.2
    rsi_upper: float = 70
    rsi_lower: float = 30

    def generate_signal(self, data: pd.DataFrame) -> Optional[StrategyResult]:
        enriched = compute_trend_indicators(data)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2]

        ema_bullish = latest["ema_20"] > latest["ema_50"]
        ema_bearish = latest["ema_20"] < latest["ema_50"]
        rsi_rebound_up = prev["rsi_14"] < self.rsi_lower and latest["rsi_14"] > self.rsi_lower
        rsi_rebound_down = prev["rsi_14"] > self.rsi_upper and latest["rsi_14"] < self.rsi_upper

        volume_increase = latest["volume"] > self.volume_threshold * latest["volume_ma"]

        if ema_bullish and rsi_rebound_up and volume_increase:
            reason = "EMA bullish crossover with RSI rebound and volume confirmation"
            return StrategyResult("buy", 0.75, reason)
        if ema_bearish and rsi_rebound_down and volume_increase:
            reason = "EMA bearish crossover with RSI pullback and volume confirmation"
            return StrategyResult("sell", 0.75, reason)
        return None
