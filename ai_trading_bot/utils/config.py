from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class DataSourceConfig:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    lookback_days: int = 365
    cache_dir: Path = Path("ai_trading_bot/data/cache")


@dataclass
class ModelConfig:
    model_name: str = "random_forest"
    params: Dict[str, object] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "random_state": 42,
    })


@dataclass
class SentimentConfig:
    enable_finbert: bool = True
    enable_vader: bool = True
    min_confidence: float = 0.55


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000.0
    commission: float = 0.00075
    risk_per_trade: float = 0.01


@dataclass
class APIConfig:
    enable_paper_trading: bool = True
    exchange: str = "binance"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


@dataclass
class TradingBotConfig:
    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    api: APIConfig = field(default_factory=APIConfig)
    enable_llm: bool = False


DEFAULT_CONFIG = TradingBotConfig()
