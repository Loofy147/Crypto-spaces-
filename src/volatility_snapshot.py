"""Validated, atomic JSON snapshot storage for the no-database deployment path."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_HISTORY_LIMIT = 720  # 30 days at an hourly cadence.


def _validate_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("generated_at must be a non-empty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("generated_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    return value


def build_snapshot_document(
    *,
    generated_at: str,
    forecasts: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> dict[str, Any]:
    """Build the stable public schema consumed by the dashboard."""
    timestamp = _validate_timestamp(generated_at)
    if not forecasts:
        raise ValueError("at least one forecast is required")
    if any(not isinstance(row, Mapping) for row in forecasts):
        raise ValueError("forecasts must contain JSON objects")
    if any(not isinstance(row, Mapping) for row in sources):
        raise ValueError("sources must contain JSON objects")
    normalized_errors = [str(error) for error in errors]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "status": "partial" if normalized_errors else "succeeded",
        "instrument": "ETH/USD",
        "forecast_interval": "1h",
        "forecast_horizon_steps": 1,
        "methodology": {
            "models": ["GARCH", "EGARCH", "GJR-GARCH"],
            "innovation_distribution": "student_t",
            "confidence_level": 0.95,
            "price_quantiles_are_conditional": True,
        },
        "sources": [dict(source) for source in sources],
        "forecasts": [dict(forecast) for forecast in forecasts],
        "errors": normalized_errors,
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def append_snapshot_history(
    path: Path,
    snapshot: Mapping[str, Any],
    *,
    max_entries: int = DEFAULT_HISTORY_LIMIT,
) -> int:
    """Append one cycle, deduplicate by timestamp and retain a bounded window.

    Returns 1 when a new cycle is recorded and 0 for an already-recorded timestamp.
    Malformed existing history raises instead of overwriting potentially useful data.
    """
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1")
    timestamp = _validate_timestamp(str(snapshot.get("generated_at", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"history contains invalid JSON on line {line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("generated_at"), str):
                raise ValueError(f"history line {line_number} has an invalid snapshot shape")
            rows.append(row)
    if any(row["generated_at"] == timestamp for row in rows):
        return 0
    rows.append(dict(snapshot))
    rows = rows[-max_entries:]
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_write(path, content)
    return 1


def write_snapshot(
    output_dir: Path,
    snapshot: Mapping[str, Any],
    *,
    max_history_entries: int = DEFAULT_HISTORY_LIMIT,
) -> Path:
    """Write latest.json and bounded history.jsonl using atomic replacements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.json"
    latest_content = json.dumps(
        dict(snapshot), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    _atomic_write(latest_path, latest_content)
    append_snapshot_history(
        output_dir / "history.jsonl", snapshot, max_entries=max_history_entries
    )
    return latest_path
