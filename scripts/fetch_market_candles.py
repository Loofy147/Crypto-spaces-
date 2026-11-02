"""Fetch canonical public spot candles and persist immutable JSONL + manifest artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.market_data_providers import CoinbaseExchangeClient, KrakenSpotClient
from src.data.ohlcv_contract import audit_candle_series


def parse_utc(value: str) -> int:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("timestamps must include a UTC offset or Z")
    return int(parsed.astimezone(timezone.utc).timestamp())


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            handle_name = handle.name
        os.replace(handle_name, path)
    finally:
        if handle_name and os.path.exists(handle_name):
            os.unlink(handle_name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("coinbase", "kraken"), required=True)
    parser.add_argument("--interval", choices=("1m", "5m", "15m", "1h", "4h", "6h", "1d"), default="1h")
    parser.add_argument("--instrument", default=None, help="Coinbase product ETH-USD or Kraken pair ETHUSD")
    parser.add_argument("--start", type=parse_utc, help="UTC ISO-8601 range start (Coinbase only)")
    parser.add_argument("--end", type=parse_utc, help="UTC ISO-8601 exclusive range end (Coinbase only)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/market-data"))
    args = parser.parse_args()

    if args.provider == "coinbase":
        if args.start is None or args.end is None:
            parser.error("Coinbase requires --start and --end")
        instrument = args.instrument or "ETH-USD"
        batch = CoinbaseExchangeClient().fetch_range(
            instrument, args.interval, args.start, args.end
        )
    else:
        if args.start is not None or args.end is not None:
            parser.error("Kraken OHLC endpoint returns a bounded recent window, not arbitrary historical ranges")
        instrument = args.instrument or "ETHUSD"
        batch = KrakenSpotClient().fetch_latest(instrument, args.interval)

    audit = audit_candle_series(batch.candles)
    if not audit.structurally_valid:
        raise SystemExit(f"series failed structural validation: {audit}")

    stamp = datetime.fromtimestamp(batch.retrieved_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{batch.provider}-{batch.returned_instrument}-{batch.interval}-{stamp}"
    data_path = args.output_dir / f"{stem}.jsonl"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    jsonl = "".join(
        json.dumps(candle.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for candle in batch.candles
    )
    manifest = batch.manifest()
    manifest["artifact"] = data_path.name
    manifest["series_audit"] = {
        "structurally_valid": audit.structurally_valid,
        "duplicate_event_times": list(audit.duplicate_event_times),
        "out_of_order_indices": list(audit.out_of_order_indices),
        "unobserved_intervals": [list(gap) for gap in audit.unobserved_intervals],
    }
    atomic_write(data_path, jsonl)
    atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    print(json.dumps({
        "data_artifact": str(data_path),
        "manifest_artifact": str(manifest_path),
        "provider": batch.provider,
        "instrument": batch.returned_instrument,
        "interval": batch.interval,
        "rows": len(batch.candles),
        "finalized_rows": sum(c.is_final for c in batch.candles),
        "unobserved_intervals": len(audit.unobserved_intervals),
        "raw_response_sha256": batch.raw_response_sha256,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
