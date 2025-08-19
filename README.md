# Barchart Swing Bot

Barchart Swing Bot is a prototype trading pipeline that ingests daily top-100 stock data, applies a deterministic analysis model, and exposes a simple API and Streamlit dashboard. The code base is intentionally light-weight and geared toward experimentation.

## Features

### Pushover notifications
The bot is configurable to send Pushover alerts for important events such as:

- pre-market gate result
- stage completion or failure
- signal armed or expired
- order sent/filled/cancelled
- trail raised
- exit executed
- risk warnings
- kill-switch toggled

Pushover credentials are supplied through the `PUSHOVER_USER` and `PUSHOVER_TOKEN` environment variables, and may be disabled by omitting them. Notification plumbing is designed to be triggered from the scheduled job stages.

### Data ingestion
`SCRAPE_ALLOWED` controls whether the bot may scrape Barchart's Top‑100 list (https://www.barchart.com/stocks/top-100-stocks). When scraping is disallowed the bot expects a CSV upload and records a normalized snapshot in the database.

### Analysis engine
A deterministic confidence model computes a percentage score from a few features. Signals are considered actionable only when the confidence is at least 80 %.

### API and scheduler
The FastAPI backend exposes health checks, a small key/value settings store and the ability to trigger scheduled jobs on demand. Additional endpoints toggle the kill‑switch, risk envelope and send test Pushover alerts to verify credentials.

### Streamlit dashboard
The `dashboard/` directory provides a minimal Streamlit front‑end that can talk to the backend. It includes buttons to run jobs immediately, toggle the kill‑switch and risk envelope, and issue a test Pushover notification.

## Usage

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server
   ```bash
   uvicorn barchart_swing_bot.main:app
   ```
3. Launch the Streamlit dashboard in another terminal
   ```bash
   streamlit run dashboard/app.py
   ```

A `docker-compose.yml` is also included for containerized development.

## Environment variables

Key configuration options are provided via environment variables:

- `PUSHOVER_USER` / `PUSHOVER_TOKEN` – credentials for Pushover notifications
- `SCRAPE_ALLOWED` – enable scraping of Barchart Top‑100 instead of CSV upload
- `RISK_ENVELOPE` – toggle the risk envelope logic
- `API_TOKEN` – token expected by protected API endpoints

See `barchart_swing_bot/config.py` for defaults and additional options.

## Testing

Run the test suite with:
```bash
pytest
```
