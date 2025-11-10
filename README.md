# AI Trading Bot

This repository provides a modular blueprint for building a multi-system AI trading bot that combines
rule-based strategies, machine-learning classification, sentiment analysis, and optional LLM
explainability.

## Features

- **Four trading subsystems** covering trend following, breakouts, mean reversion, and orderflow-based
  liquidity sweeps.
- **Machine-learning engine** that engineers features from price/volume data and trains a Random Forest
  classifier for signal confirmation.
- **Sentiment analysis** via FinBERT and VADER as optional filters.
- **Backtesting integration** using Backtrader for unified strategy evaluation.
- **FastAPI server stub** to expose bot configuration and strategy metadata for live or paper trading
  setups.
- **Optional LLM hooks** (LangChain compatible) for trade explanations, drawdown diagnostics, and
  strategy advisory.

## Project Layout

```
ai_trading_bot/
├── api/                    # FastAPI server entry points
├── backtest/               # Backtesting runners
├── indicators/             # Technical & volume indicators
├── llm_module/             # Optional local LLM helpers
├── ml_engine/              # Feature engineering & model training
├── sentiment/              # Sentiment analysis modules
├── strategies/             # Trading strategy implementations
└── utils/                  # Shared utilities (config, logging, data loading)
```

## Getting Started



## Configuration

Adjust runtime options in `ai_trading_bot/utils/config.py`. The `TradingBotConfig` dataclass controls
symbols, intervals, ML parameters, sentiment preferences, and backtesting risk settings.

## API Server

```bash
uvicorn ai_trading_bot.api.server:create_app --factory
```

The API exposes:

- `GET /config` – Returns the current configuration.
- `GET /strategies` – Describes strategy parameters for transparency dashboards.

## License

This project is distributed under the MIT license.
