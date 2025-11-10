from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

from .config import DataSourceConfig
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataLoader:
    config: DataSourceConfig

    def _get_cache_path(self) -> Path:
        cache_dir = self.config.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.config.symbol}_{self.config.interval}.parquet"
        return cache_dir / filename

    def load_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        cache_path = self._get_cache_path()
        if cache_path.exists() and not force_refresh:
            logger.info("Loading cached data from %s", cache_path)
            return pd.read_parquet(cache_path)

        if yf is None:
            raise ImportError(
                "yfinance is required to download data. Please install it via `pip install yfinance`."
            )

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        logger.info(
            "Downloading historical data for %s from %s to %s",
            self.config.symbol,
            start_date,
            end_date,
        )
        ticker = yf.Ticker(self.config.symbol)
        df = ticker.history(start=start_date, end=end_date, interval=self._map_interval())
        if df.empty:
            raise ValueError("No data returned from yfinance. Check symbol and interval.")

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df.index.name = "timestamp"
        df.to_parquet(cache_path)
        return df

    def _map_interval(self) -> str:
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "4h",
            "1d": "1d",
        }
        interval = mapping.get(self.config.interval)
        if interval is None:
            raise ValueError(f"Unsupported interval: {self.config.interval}")
        return interval

    def load_live_data(self) -> pd.DataFrame:
        logger.warning("Live data loading is not implemented in the offline prototype.")
        return pd.DataFrame()
