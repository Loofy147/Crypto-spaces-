"""Tests for the rolling ETH volatility benchmark helpers."""

import numpy as np
import pytest

from scripts.evaluate_eth_volatility import (
    _daily_realized_series,
    _dm_qlike,
    _holm_adjust,
    _qlike,
    downside_har_variance_forecast,
)


def test_daily_realized_series_keeps_only_complete_utc_days():
    day_start = 1_800_000_000 - (1_800_000_000 % 86_400)
    times = list(range(day_start + 3_600, day_start + 86_401, 3_600))
    values = np.linspace(-0.01, 0.012, 24)
    # Add an incomplete second day; it must not become a realized-volatility row.
    times.extend([day_start + 86_400 + 3_600, day_start + 86_400 + 7_200])
    values_all = np.concatenate([values, np.asarray([0.02, -0.03])])

    day_ends, realized, downside = _daily_realized_series(values_all, times)

    assert day_ends == [day_start + 86_400]
    assert realized[0] == pytest.approx(float(np.dot(values, values)))
    negative = np.minimum(values, 0.0)
    assert downside[0] == pytest.approx(float(np.dot(negative, negative)))


def test_daily_realized_series_rejects_unaligned_or_duplicate_hours():
    day_start = 1_800_000_000 - (1_800_000_000 % 86_400)
    times = list(range(day_start + 3_600, day_start + 86_401, 3_600))
    times[-1] = times[-2]
    _, realized, _ = _daily_realized_series(np.full(24, 0.01), times)
    assert realized.size == 0


def test_downside_har_forecast_is_finite_and_positive():
    rng = np.random.default_rng(2026)
    rv = np.exp(rng.normal(-8.5, 0.3, 120))
    downside = rv * rng.uniform(0.25, 0.7, 120)
    downside[::13] = 0.0

    forecast = downside_har_variance_forecast(rv, downside)

    assert np.isfinite(forecast)
    assert forecast > 0.0


def test_downside_har_rejects_misaligned_data():
    with pytest.raises(ValueError, match="aligned"):
        downside_har_variance_forecast(np.ones(60), np.ones(59))


def test_qlike_and_dm_output_are_finite_for_aligned_forecasts():
    actual = np.asarray([0.8, 1.2, 1.1, 0.9, 1.4, 1.0], dtype=np.float64)
    benchmark = np.asarray([0.9, 1.0, 1.0, 1.0, 1.2, 1.0], dtype=np.float64)
    candidate = np.asarray([0.85, 1.1, 1.0, 0.95, 1.3, 1.0], dtype=np.float64)

    losses = _qlike(actual, candidate)
    result = _dm_qlike(actual, candidate, benchmark, hac_lags=2)

    assert losses.shape == actual.shape
    assert np.all(np.isfinite(losses))
    assert result["observations"] == actual.size
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["hac_lags"] == 2


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = _holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.2)
