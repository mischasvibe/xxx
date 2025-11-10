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

> **Python version**: The full dependency stack currently supports Python 3.10–3.12. Newer releases
> (3.13+) work for the core modules, but several optional ML libraries do not yet publish wheels.
> On macOS, ensure that the `python3` command points to a supported version (e.g. via Homebrew
> `python@3.11`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ai_trading_bot.main
```

If you are constrained to Python 3.13 or newer, `pip` automatically skips the optional packages that
have not shipped compatible wheels (`torch`, `tensorflow`, `prophet`, `stable-baselines3`). The bot
will still run with the core rule-based strategies, but advanced ML and forecasting features remain
disabled until those libraries add support. On supported Python versions you can install the full
stack with the same command.

By default the bot downloads one year of hourly BTC/USDT data via `yfinance`. Optional components
(FinBERT, Prophet, LangChain, Backtrader, etc.) rely on the dependencies declared in
`requirements.txt` and may require additional system packages as described in their respective
documentation.

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
