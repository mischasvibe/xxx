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

> **Python version**: Use Python **3.10–3.12**. Core dependencies such as `scikit-learn` currently do
> not publish wheels for 3.13+ or 3.14, so installation on newer interpreters fails. On macOS, link
> `python3` to a supported version (for example Homebrew `python@3.11`) before creating the virtual
> environment.

### 1. Verify your interpreter

Make sure the `python3` on your `PATH` points to a supported version:

```bash
python3 --version
```

If the output is **3.13** or newer, explicitly invoke the supported interpreter when creating the
virtual environment, for example `/usr/local/bin/python3.11` on macOS Homebrew installs.

### 2. Create the virtual environment

Run each command below **one at a time** in your shell. Copying the whole block at once can trigger
errors in `zsh`, so prefer pasting and executing line-by-line:

```bash
# (Optional) remove any previously created virtual environment
rm -rf .venv

python3.11 -m venv .venv  # replace with the supported interpreter path on your system
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ai_trading_bot.main
```

The default requirements install only the cross-platform dependencies so the core bot runs on
CPython 3.10–3.12. Optional, heavy ML packages remain in `requirements-ml.txt`. Install them only
when using a supported Python (3.10–3.12) or when wheels are available for your platform:

```bash
python -m pip install -r requirements-ml.txt
```

This adds `prophet`, `stable-baselines3`, `torch`, `tensorflow`, and `transformers` back into the
environment for full machine-learning, forecasting, and FinBERT sentiment support.

### Troubleshooting install issues

- **`SystemExit: This project currently supports Python versions between 3.10 (inclusive) and 3.13 (exclusive)`** – Switch to
  Python 3.10–3.12 and recreate the virtual environment. On macOS you can install Homebrew
  `python@3.11` and run `python3.11 -m venv .venv`.
- **`ModuleNotFoundError: No module named 'pandas'`** – Dependency installation stopped early,
  typically because an unsupported Python version was used. Remove `.venv`, ensure you are using
  Python 3.10–3.12, upgrade `pip`, and reinstall via the commands above.
- **`Could not find a version that satisfies the requirement torch`** – Torch does not ship wheels for
  newer interpreters yet. Skip `requirements-ml.txt` or downgrade to Python 3.10–3.12. FinBERT
  sentiment falls back to VADER automatically when the optional dependencies are missing.

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
