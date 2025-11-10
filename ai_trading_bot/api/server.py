from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from fastapi import FastAPI

from ..strategies.breakout import BreakoutStrategy
from ..strategies.orderflow import OrderflowStrategy
from ..strategies.reversion import MeanReversionStrategy
from ..strategies.trend_following import TrendFollowingStrategy
from ..utils.config import TradingBotConfig


@dataclass
class BotState:
    config: TradingBotConfig


def create_app(config: TradingBotConfig) -> FastAPI:
    app = FastAPI(title="AI Trading Bot")
    state = BotState(config=config)

    @app.get("/config")
    def get_config() -> Dict[str, object]:
        return asdict(state.config)

    @app.get("/strategies")
    def list_strategies() -> Dict[str, Dict[str, float]]:
        strategies = [
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            MeanReversionStrategy(),
            OrderflowStrategy(),
        ]
        return {strat.name: strat.__dict__ for strat in strategies}

    return app
