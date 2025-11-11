from __future__ import annotations

import pandas as pd


def delta_volume(df: pd.DataFrame) -> pd.Series:
    signed_volume = (df["close"] - df["open"]).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (signed_volume * df["volume"]).cumsum()


def fair_value_gap(df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    previous_high = df["high"].shift(1)
    previous_low = df["low"].shift(1)
    gap = previous_low - df["high"]
    imbalance = gap.where(gap > 0)
    return imbalance.rolling(window=lookback).max()


def compute_orderflow_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["delta_volume"] = delta_volume(data)
    data["fair_value_gap"] = fair_value_gap(data)
    data["rolling_volume"] = data["volume"].rolling(window=10).mean()
    return data
