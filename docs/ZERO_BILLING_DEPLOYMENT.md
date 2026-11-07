# Zero-billing ETH/USD volatility deployment

## Current deployment target

The Render Blueprint was removed from this branch because it provisions billable services. The initial no-card operating path is:

- **GitHub Actions:** hourly public market-data retrieval and model fitting.
- **Versioned repository artifacts:** latest snapshot plus a bounded 30-day JSONL history.
- **Vercel Hobby:** dashboard and a read-only Next.js API route that serves the latest public snapshot.
- **No database or runtime API server is required for the first release.** The existing PostgreSQL schema, repository and FastAPI service remain available for a later isolated Neon/Supabase database migration.

## Snapshot contract

- Path: `artifacts/volatility/latest.json`
- History: `artifacts/volatility/history.jsonl`
- Schema: `schema_version = 1`
- Instrument: ETH/USD
- Candle interval and forecast horizon: 1 hour
- Models: GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1)
- Innovation distribution: standardized Student-t
- Forecast interval: conditional 95% price quantiles
- Sources remain separate: Coinbase Exchange and Kraken Spot are never blended.
- Inputs are the latest finalized contiguous candle segment. Missing intervals are not filled or bridged.
- History retains at most 720 hourly snapshots (30 days). Each snapshot records source manifests, data hashes, model statistics and forecast values.

## Scheduling and reliability

The GitHub Actions workflow runs at minute 17 of each UTC hour and can also be dispatched manually. GitHub may delay or drop scheduled runs during periods of high load. The dashboard must therefore display the snapshot generation time and mark data older than 90 minutes as stale. A failed cycle must not overwrite the previous valid snapshot.

The workflow writes only to the repository that contains the public snapshot. It requires `contents: write` for its built-in `GITHUB_TOKEN`; no exchange API key, database URL, Vercel token or other secret is required. Repository settings must allow GitHub Actions to write contents.

## Release gates

1. Snapshot-storage tests pass.
2. The scheduled workflow is merged into the default branch; GitHub schedules only workflows on the default branch.
3. A manual workflow run creates `latest.json` and `history.jsonl`.
4. The output is validated against schema version 1 and contains forecasts from at least one venue.
5. The Vercel dashboard preview reads the snapshot endpoint and displays its real generated timestamp.
6. The production deployment is promoted only after the preview is verified end to end.

## Deferred infrastructure

PostgreSQL persistence and the FastAPI database API are not claimed as live. They remain a future upgrade path once a dedicated no-card database can be provisioned. Do not connect this project to an existing unrelated database or expose database credentials to the browser.

## Interpretation

Forecasts describe conditional volatility and conditional price quantiles, not price direction or guaranteed trading levels. They are research outputs, not execution instructions. Rolling out-of-sample interval coverage and proper variance scores remain required before describing the forecasts as calibrated.
