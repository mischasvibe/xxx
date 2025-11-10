from __future__ import annotations

from typing import List

import pandas as pd

from ..indicators.bollinger_rsi import compute_reversion_indicators
from ..indicators.ema_rsi_volume import compute_trend_indicators
from ..indicators.orderflow_liquidity import compute_orderflow_indicators
from ..indicators.vwap_range import compute_breakout_indicators


FEATURE_COLUMNS = [
    "close",
    "volume",
    "ema_20",
    "ema_50",
    "rsi_14",
    "obv",
    "vwap",
    "range_high",
    "range_low",
    "bb_upper",
    "bb_lower",
    "delta_volume",
    "fair_value_gap",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    trend = compute_trend_indicators(df)
    breakout = compute_breakout_indicators(df)
    reversion = compute_reversion_indicators(df)
    orderflow = compute_orderflow_indicators(df)

    combined = trend.join(
        [
            breakout[["vwap", "range_high", "range_low"]],
            reversion[["bb_upper", "bb_lower"]],
            orderflow[["delta_volume", "fair_value_gap"]],
        ]
    )
    combined = combined.dropna().copy()
    return combined


def create_labels(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.002) -> pd.Series:
    future_return = df["close"].pct_change(periods=horizon).shift(-horizon)
    labels = future_return.apply(
        lambda r: "buy" if r > threshold else ("sell" if r < -threshold else "hold")
    )
    return labels


def select_feature_columns(data: pd.DataFrame, columns: List[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    available = [c for c in columns if c in data.columns]
    return data[available]
