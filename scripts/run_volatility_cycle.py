"""Fetch completed ETH-USD spot candles and persist per-venue model forecasts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone

from src.data.market_data_providers import CoinbaseExchangeClient, KrakenSpotClient
from src.data.ohlcv_contract import (
    audit_candle_series,
    finalized_contiguous_log_returns,
    longest_contiguous_finalized_segment,
)
from src.models.volatility_forecasting import (
    fit_volatility_model,
    forecast_price_interval,
    next_period_variance,
)
from src.storage.postgres_repository import PostgresRepository


def _utc_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def _forecast_payload(venue: str, candles: tuple, model: str) -> dict:
    finalized = longest_contiguous_finalized_segment(candles)
    if len(finalized) < 100:
        raise ValueError(f"{venue}: only {len(finalized)} contiguous finalized bars; need at least 100")
    returns = finalized_contiguous_log_returns(finalized)
    fit = fit_volatility_model(returns, model=model, distribution="student_t")
    spot = finalized[-1].close
    interval = forecast_price_interval(
        fit, spot_price=spot, confidence=0.95, periods_per_year=365 * 24
    )
    data_hash = hashlib.sha256(
        json.dumps(
            [c.to_dict() for c in finalized],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "venue": venue,
        "instrument": finalized[-1].instrument,
        "base_currency": finalized[-1].base_currency,
        "quote_currency": finalized[-1].quote_currency,
        "interval": finalized[-1].interval,
        "forecast_horizon_steps": 1,
        "forecast_origin": finalized[-1].event_time,
        "model": model,
        "innovation_distribution": fit.distribution,
        "observations": fit.observations,
        "input_sha256": data_hash,
        "model_parameters": dict(fit.parameters),
        "log_likelihood": fit.log_likelihood,
        "aic": fit.aic,
        "bic": fit.bic,
        "conditional_variance": next_period_variance(fit),
        "conditional_volatility": interval.conditional_volatility,
        "annualized_volatility": interval.annualized_volatility,
        "reference_price": spot,
        "mean_log_return": interval.mean_log_return,
        "degrees_of_freedom": interval.degrees_of_freedom,
        "price_quantiles": {
            "confidence": interval.confidence,
            "lower": interval.lower_price,
            "median": interval.median_price,
            "upper": interval.upper_price,
        },
    }


def main() -> int:
    now = _utc_epoch()
    # Coinbase provides a historical range; 120 days gives the hourly models
    # adequate data while keeping API pagination bounded.
    end = now - (now % 3600)
    start = end - 120 * 24 * 3600
    coinbase = CoinbaseExchangeClient().fetch_range(
        "ETH-USD", "1h", start, end, as_of_timestamp=now
    )
    # Kraken is a recent independent comparison feed; its REST endpoint returns
    # at most 720 rows and includes an unfinished current candle.
    kraken = KrakenSpotClient().fetch_latest(
        "ETHUSD", "1h", as_of_timestamp=now
    )
    batches = (coinbase, kraken)
    cycle_errors: list[str] = []
    forecasts_created = 0
    with PostgresRepository.from_dsn() as repository:
        cycle_id = repository.start_forecast_cycle()
        for batch in batches:
            audit = audit_candle_series(batch.candles)
            if not audit.structurally_valid:
                cycle_errors.append(f"{batch.provider}: mixed, duplicate or unordered candle series")
                continue
            run_id = repository.start_market_data_run(batch.manifest())
            try:
                repository.upsert_candles(batch.candles)
                repository.finish_market_data_run(
                    run_id, status="succeeded", row_count=len(batch.candles),
                    finalized_count=sum(c.is_final for c in batch.candles),
                    raw_response_sha256=batch.raw_response_sha256,
                )
            except Exception as exc:
                repository.finish_market_data_run(
                    run_id, status="failed", error_message=str(exc)
                )
                cycle_errors.append(f"{batch.provider} ingestion: {exc}")
                continue
            for model in ("GARCH", "EGARCH", "GJR-GARCH"):
                try:
                    payload = _forecast_payload(batch.provider, batch.candles, model)
                    repository.save_forecast(payload)
                    forecasts_created += 1
                except Exception as exc:
                    cycle_errors.append(f"{batch.provider}/{model}: {exc}")
        status = "failed" if forecasts_created == 0 else ("partial" if cycle_errors else "succeeded")
        repository.finish_forecast_cycle(
            cycle_id, status=status, forecasts_created=forecasts_created,
            errors=cycle_errors,
        )
    print(json.dumps({
        "status": "failed" if forecasts_created == 0 else ("partial" if cycle_errors else "succeeded"),
        "forecasts_created": forecasts_created,
        "errors": cycle_errors,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return 1 if forecasts_created == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
