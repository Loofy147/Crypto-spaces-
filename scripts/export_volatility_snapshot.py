"""Fetch ETH/USD public candles and publish a bounded, database-free forecast snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.market_data_providers import CoinbaseExchangeClient, KrakenSpotClient
from src.data.ohlcv_contract import audit_candle_series, longest_contiguous_finalized_segment
from src.volatility_snapshot import build_snapshot_document, write_snapshot
from scripts.run_volatility_cycle import _forecast_payload

MODELS = ("GARCH", "EGARCH", "GJR-GARCH")


def collect_snapshot(output_dir: Path) -> dict:
    """Fetch each venue independently; one provider failure does not hide the other."""
    now = int(datetime.now(timezone.utc).timestamp())
    end = now - (now % 3600)
    start = end - 120 * 24 * 3600
    batches = []
    errors: list[str] = []

    try:
        batches.append(CoinbaseExchangeClient().fetch_range(
            "ETH-USD", "1h", start, end, as_of_timestamp=now
        ))
    except Exception as exc:
        errors.append(f"coinbase fetch: {type(exc).__name__}: {exc}")

    try:
        batches.append(KrakenSpotClient().fetch_latest(
            "ETHUSD", "1h", as_of_timestamp=now
        ))
    except Exception as exc:
        errors.append(f"kraken fetch: {type(exc).__name__}: {exc}")

    sources: list[dict] = []
    forecasts: list[dict] = []
    for batch in batches:
        audit = audit_candle_series(batch.candles)
        source = batch.manifest()
        source["series_audit"] = {
            "structurally_valid": audit.structurally_valid,
            "duplicate_event_times": list(audit.duplicate_event_times),
            "out_of_order_indices": list(audit.out_of_order_indices),
            "unobserved_intervals": [list(gap) for gap in audit.unobserved_intervals],
        }
        if not audit.structurally_valid:
            source["status"] = "rejected"
            errors.append(f"{batch.provider}: mixed, duplicate or unordered candle series")
            sources.append(source)
            continue

        finalized = longest_contiguous_finalized_segment(batch.candles)
        source["status"] = "succeeded"
        source["latest_finalized_candle"] = finalized[-1].to_dict() if finalized else None
        source["price_history"] = [
            {
                "event_time": candle.event_time,
                "close": candle.close,
                "high": candle.high,
                "low": candle.low,
                "volume": candle.volume,
            }
            for candle in finalized[-72:]
        ]
        sources.append(source)

        for model in MODELS:
            try:
                forecasts.append(_forecast_payload(batch.provider, batch.candles, model))
            except Exception as exc:
                errors.append(f"{batch.provider}/{model}: {type(exc).__name__}: {exc}")

    if not forecasts:
        raise RuntimeError(
            "no valid forecasts produced; preserving the previous snapshot: " + "; ".join(errors)
        )

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    snapshot = build_snapshot_document(
        generated_at=generated_at,
        forecasts=forecasts,
        sources=sources,
        errors=errors,
    )
    snapshot["generated_at_epoch"] = int(datetime.now(timezone.utc).timestamp())
    latest_path = write_snapshot(output_dir, snapshot)
    print(json.dumps({
        "status": snapshot["status"],
        "generated_at": snapshot["generated_at"],
        "source_count": len(sources),
        "forecast_count": len(forecasts),
        "errors": errors,
        "latest_snapshot": str(latest_path),
    }, indent=2))
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/volatility"))
    args = parser.parse_args()
    collect_snapshot(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
