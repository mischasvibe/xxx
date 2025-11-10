from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..indicators.vwap_range import compute_breakout_indicators
from .base import Strategy, StrategyResult


@dataclass
class BreakoutStrategy(Strategy):
    name: str = "breakout"
    volume_multiplier: float = 1.3

    def generate_signal(self, data: pd.DataFrame) -> Optional[StrategyResult]:
        enriched = compute_breakout_indicators(data)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2]

        breakout_up = latest["close"] > latest["range_high"] and latest["close"] > latest["vwap"]
        breakout_down = latest["close"] < latest["range_low"] and latest["close"] < latest["vwap"]
        volume_spike = latest["volume"] > self.volume_multiplier * latest["volume_ma"]

        if breakout_up and volume_spike:
            reason = "Price breaks above range high and VWAP with volume confirmation"
            return StrategyResult("buy", 0.7, reason)
        if breakout_down and volume_spike:
            reason = "Price breaks below range low and VWAP with volume confirmation"
            return StrategyResult("sell", 0.7, reason)
        return None
