from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..indicators.orderflow_liquidity import compute_orderflow_indicators
from .base import Strategy, StrategyResult


@dataclass
class OrderflowStrategy(Strategy):
    name: str = "orderflow"
    delta_threshold: float = 0.0

    def generate_signal(self, data: pd.DataFrame) -> Optional[StrategyResult]:
        enriched = compute_orderflow_indicators(data)
        latest = enriched.iloc[-1]
        prev = enriched.iloc[-2]

        absorption_buy = (
            latest["low"] < prev["low"]
            and latest["close"] > prev["close"]
            and latest["delta_volume"] > prev["delta_volume"] + abs(prev["delta_volume"]) * 0.05
        )
        absorption_sell = (
            latest["high"] > prev["high"]
            and latest["close"] < prev["close"]
            and latest["delta_volume"] < prev["delta_volume"] - abs(prev["delta_volume"]) * 0.05
        )

        if absorption_buy:
            reason = "Liquidity sweep with delta volume absorption on the bid"
            return StrategyResult("buy", 0.72, reason)
        if absorption_sell:
            reason = "Liquidity sweep with delta volume absorption on the ask"
            return StrategyResult("sell", 0.72, reason)
        return None
