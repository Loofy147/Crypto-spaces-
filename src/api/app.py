"""Read-only HTTP API for persisted ETH volatility forecasts."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.storage.postgres_repository import PostgresRepository


app_logger = logging.getLogger("eth-volatility-api")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _open_repository() -> PostgresRepository:
    try:
        return PostgresRepository.from_dsn()
    except Exception as exc:
        app_logger.exception("Could not connect to the forecast database")
        raise HTTPException(status_code=503, detail="database unavailable") from exc


app = FastAPI(
    title="ETH Volatility Intelligence API",
    version="0.1.0",
    description=(
        "Read-only API for versioned, per-venue conditional-volatility forecasts. "
        "Forecasts are research outputs, not execution instructions."
    ),
)

cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["Content-Type"],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness only; it does not claim database readiness."""
    return {
        "status": "ok",
        "service": "eth-volatility-api",
        "database_configured": bool(os.environ.get("DATABASE_URL")),
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness check verifies the configured PostgreSQL connection."""
    repository = _open_repository()
    try:
        if not repository.ping():
            raise HTTPException(status_code=503, detail="database ping failed")
        return {"status": "ready", "database": "connected"}
    finally:
        repository.close()


@app.get("/v1/forecasts/latest")
def get_latest_forecasts(
    base_currency: str = Query(default="ETH", min_length=2, max_length=16),
    quote_currency: str = Query(default="USD", min_length=2, max_length=16),
    interval: str = Query(default="1h", pattern="^(1h|4h|1d)$"),
    venue: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return the latest persisted model forecasts; never compute on request."""
    repository = _open_repository()
    try:
        rows = repository.latest_forecasts(
            base_currency=base_currency, quote_currency=quote_currency,
            interval=interval, venue=venue, limit=limit
        )
        return {
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "interval": interval,
            "venue_filter": venue,
            "count": len(rows),
            "forecasts": _json_safe(rows),
        }
    finally:
        repository.close()


@app.get("/v1/market-data/status")
def get_market_data_status(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    repository = _open_repository()
    try:
        rows = repository.latest_market_data_runs(limit=limit)
        return {"count": len(rows), "sources": _json_safe(rows)}
    finally:
        repository.close()
