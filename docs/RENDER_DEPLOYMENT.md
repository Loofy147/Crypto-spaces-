# Render deployment readiness

## Service topology

The root \`render.yaml\` describes three billable resources in the Frankfurt
region:

- \`eth-volatility-api\`: read-only FastAPI service. Liveness endpoint
  \`/health\`; database readiness endpoint \`/ready\`.
- \`eth-volatility-hourly\`: hourly scheduled ingestion and model cycle.
- \`eth-volatility-db\`: PostgreSQL persistence for candles, ingestion runs,
  forecast cycles and forecast records.

The web API reads persisted forecasts. It does not fit models during requests.
The cron job fetches completed Coinbase and Kraken spot candles independently,
persists source observations and fits GARCH-family candidates per venue.

## Deployment contract

Render installs \`requirements-api.txt\` and starts the API with:

\`\`\`sh
uvicorn src.api.app:app --host 0.0.0.0 --port $PORT
\`\`\`

The scheduled worker command is:

\`\`\`sh
python scripts/run_volatility_cycle.py
\`\`\`

The Blueprint binds \`DATABASE_URL\` from the managed PostgreSQL instance.
\`CORS_ORIGINS\` is intentionally declared with \`sync: false\`; populate it
with the exact deployed dashboard origin(s), comma-separated. Do not use a
wildcard when credentials or private data are introduced.

## Before creating resources

1. Review the plans and current prices in Render. Blueprint sync creates
   billable compute, cron and PostgreSQL resources.
2. Confirm the Frankfurt region and database plan are acceptable.
3. Sync the Blueprint from the feature branch, not the default branch.
4. Wait for the database and API deploys to complete.
5. Confirm \`GET /health\` returns liveness and \`GET /ready\` verifies DB access.
6. Trigger one cron run manually; inspect its logs and database run records.
7. Confirm both venues have recent completed candles and that forecast records
   carry distinct venue identifiers and source-data hashes.
8. Configure the dashboard's API base URL and CORS origin only after the API
   URL is known.

## Operational cautions

- This is an hourly forecast cycle, not a streaming tick feed.
- Kraken's public OHLC endpoint is limited to its recent window; Coinbase
  provides bounded historical pagination. Their histories will not be equal.
- Forecasts are conditional distribution estimates, not guaranteed price
  boundaries or trade instructions.
- The first deploy is a research/staging environment. Do not represent model
  accuracy as established until rolling out-of-sample calibration and coverage
  reports are available.
- Database retention, backups, migrations, alerts, API authentication/rate
  limits and production incident procedures remain separate release gates.

## Evidence checklist

| Claim | Required evidence |
|---|---|
| API deployed | Successful Render deploy + public health response |
| Database ready | \`/ready\` + successful PostgreSQL ping |
| Ingestion works | Successful provider run records + persisted candles |
| Forecast cycle works | Successful cycle + forecast rows |
| Forecast is calibrated | Rolling-origin coverage and loss report |
| Dashboard is live | Deployed UI reads API data, not fixtures |
