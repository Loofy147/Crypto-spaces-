"""Tests for the database-free forecast snapshot contract."""

from pathlib import Path
import json

import pytest

from src.volatility_snapshot import (
    append_snapshot_history,
    build_snapshot_document,
    write_snapshot,
)


def sample_snapshot(timestamp: str, price: float = 3000.0) -> dict:
    return build_snapshot_document(
        generated_at=timestamp,
        forecasts=[{"venue": "coinbase", "model": "GARCH", "reference_price": price}],
        sources=[{"provider": "coinbase", "structurally_valid": True}],
        errors=[],
    )


def test_snapshot_document_has_explicit_schema_and_status() -> None:
    snapshot = sample_snapshot("2026-11-02T15:00:00Z")
    assert snapshot["schema_version"] == 1
    assert snapshot["status"] == "succeeded"
    assert snapshot["instrument"] == "ETH/USD"
    assert snapshot["forecast_interval"] == "1h"


def test_snapshot_document_rejects_empty_forecasts() -> None:
    with pytest.raises(ValueError, match="at least one forecast"):
        build_snapshot_document(
            generated_at="2026-11-02T15:00:00Z", forecasts=[], sources=[], errors=[]
        )


def test_history_append_is_idempotent_and_bounded(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    for hour in range(4):
        snapshot = sample_snapshot(f"2026-11-02T{hour:02d}:00:00Z", 3000 + hour)
        assert append_snapshot_history(history, snapshot, max_entries=3) == 1
    duplicate = sample_snapshot("2026-11-02T03:00:00Z", 9999)
    assert append_snapshot_history(history, duplicate, max_entries=3) == 0

    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert len(rows) == 3
    assert rows[-1]["generated_at"] == "2026-11-02T03:00:00Z"
    assert rows[-1]["forecasts"][0]["reference_price"] == 3003


def test_write_snapshot_atomically_writes_latest_and_history(tmp_path: Path) -> None:
    snapshot = sample_snapshot("2026-11-02T15:00:00Z")
    latest = write_snapshot(tmp_path, snapshot)
    assert latest == tmp_path / "latest.json"
    assert json.loads(latest.read_text()) == snapshot
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 1


def test_timestamp_requires_timezone() -> None:
    with pytest.raises(ValueError, match="include a timezone"):
        sample_snapshot("2026-11-02T15:00:00")
