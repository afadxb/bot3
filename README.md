# Propulsion Bot (Phase 2)

Propulsion Bot Phase 2 is an intraday trading automation system that connects to
Interactive Brokers (IBKR), ingests market data, computes technical
indicators/features, evaluates trading strategies, and manages live orders with
risk controls. The codebase was refactored from a monolithic script into a
modular package to simplify maintenance and future enhancements.

## Project layout

```
propulsion_bot_phase2.py  # CLI entry point that runs the intraday orchestrator
propulsion_bot/
├── __init__.py           # Exposes the orchestrator factory
├── config.py             # Settings loader backed by config/strategy.yaml and .env
├── dtos.py               # Dataclasses describing domain objects (bars, signals, orders)
├── database.py           # SQLite persistence utilities for historical and live data
├── features.py           # Feature/indicator engineering pipeline
├── indicators.py         # Custom indicator implementations
├── market.py             # Market schedule helpers and trading day checks
├── strategy.py           # Strategy selection and signal generation logic
├── risk.py               # Portfolio and order risk management checks
├── trade.py              # Order routing and execution bookkeeping
├── ibkr.py               # Interactive Brokers client wrappers and callbacks
└── orchestrator.py       # Intraday orchestration loop tying all modules together
config/strategy.yaml      # Strategy configuration loaded at runtime (example)
propulsion_bot.db         # Default SQLite database created at runtime
```

## Getting started

1. **Create a Python environment** (Python 3.10+ recommended) and install
   project dependencies. The modules rely on common scientific and brokerage
   libraries such as `pandas`, `numpy`, `ib_insync`, `sqlalchemy`, and
   `python-dotenv`.
2. **Configure credentials** by creating a `.env` file in the project root (or
   exporting environment variables) with values for:
   - `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
   - `FINNHUB_TOKEN`
   - `DASHBOARD_PORT`
   - `PREMARKET_DB`
   - `FORCE_MARKET_OPEN`
3. **Tune strategy settings** by editing `config/strategy.yaml`. The
   `propulsion_bot.config.Settings` singleton exposes nested configuration paths
   via `Settings().get("section.option")`.
4. **Prepare the database**. By default the bot creates/uses `propulsion_bot.db`
   in the repository root. You may point to an existing SQLite file by setting
   the `DB_PATH` key inside the strategy configuration.

## Running the bot

Launch the intraday orchestrator via the entry point script:

```bash
python propulsion_bot_phase2.py
```

This prints banner information, instantiates the orchestrator, and begins the
continuous trading loop. The process listens for a `KeyboardInterrupt`
(`Ctrl+C`) to trigger graceful shutdown, including closing any active IBKR
connections.

## Module overview

- **Configuration (`propulsion_bot.config`)** – Centralized runtime settings
  loader sourcing YAML configuration and environment variables.
- **Data transfer objects (`propulsion_bot.dtos`)** – Dataclasses representing
  time-series bars, features, signals, orders, and IBKR events for type-safe
  communication between modules.
- **Persistence (`propulsion_bot.database`)** – Database helpers for reading and
  writing historical, intraday, and analytics tables backed by SQLite.
- **Feature engineering (`propulsion_bot.features` & `propulsion_bot.indicators`)** –
  Functions that compute derived indicators, enrich raw market data, and cache
  intermediate results.
- **Strategy (`propulsion_bot.strategy`)** – Core trading logic combining
  features into actionable signals with position sizing heuristics.
- **Risk management (`propulsion_bot.risk`)** – Pre-trade and post-trade checks
  enforcing exposure, drawdown, and liquidity constraints.
- **Trading (`propulsion_bot.trade`)** – Order staging, execution tracking, and
  reconciliation utilities for IBKR.
- **IBKR integration (`propulsion_bot.ibkr`)** – Convenience layer on top of
  `ib_insync` providing connection management and event callbacks.
- **Intraday orchestration (`propulsion_bot.orchestrator`)** – Scheduler that
  coordinates data refresh, signal generation, and order submission while
  respecting market hours.

## Development workflow

- Run static compilation checks before committing:

  ```bash
  python -m compileall propulsion_bot_phase2.py propulsion_bot
  ```

- Add new modules within the `propulsion_bot` package to keep responsibilities
  well separated. Update this README when additional components are introduced.

## Support

For questions or enhancements, create an issue describing the desired behavior
or bug. Include logs from the `logs/` directory (if enabled in your
configuration) and note the configuration values relevant to the report.
