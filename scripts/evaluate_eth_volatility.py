"""Rolling out-of-sample ETH/USD volatility benchmark.

Forecast origins are UTC day boundaries. Every candidate is scored against the
same next-24-hour realized variance target. This is a research screen, not a
production model selector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.market_data_providers import CoinbaseExchangeClient
from src.data.ohlcv_contract import (
    audit_candle_series,
    finalized_contiguous_log_returns,
    longest_contiguous_finalized_segment,
)
from src.models.horizon_forecasting import forecast_price_interval_horizon
from src.models.volatility_forecasting import (
    ewma_variance_forecast,
    fit_volatility_model,
    har_rv_forecast,
)

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24
VARIANCE_CONFIG = (
    ("GARCH", "student_t", "GARCH"),
    ("EGARCH", "normal", "EGARCH-normal"),
    ("GJR-GARCH", "student_t", "GJR-GARCH"),
)
INTERVAL_CONFIG = (
    ("GARCH", "student_t", "GARCH"),
    ("EGARCH", "student_t", "EGARCH-student_t"),
    ("GJR-GARCH", "student_t", "GJR-GARCH"),
)
INTERVAL_NAMES = tuple(item[2] for item in INTERVAL_CONFIG)
MODEL_NAMES = (
    "GARCH", "EGARCH-normal", "GJR-GARCH", "HAR-RV",
    "HAR-RV+downside", "EWMA-hourly", "Rolling-RV-22d",
)


def _daily_realized_series(
    returns: Sequence[float], end_times: Sequence[int]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return complete UTC-day RV and downside semivariance observations."""
    if len(returns) != len(end_times):
        raise ValueError("returns and end_times must have the same length")
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for value, event_time in zip(returns, end_times):
        if not math.isfinite(float(value)) or event_time <= 0:
            raise ValueError("daily aggregation received an invalid observation")
        day_start = ((int(event_time) - 1) // SECONDS_PER_DAY) * SECONDS_PER_DAY
        grouped[day_start].append((int(event_time), float(value)))

    day_ends: list[int] = []
    realized: list[float] = []
    downside: list[float] = []
    for day_start, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[0])
        expected_times = list(range(
            day_start + 3_600, day_start + SECONDS_PER_DAY + 1, 3_600
        ))
        if [row[0] for row in rows] != expected_times:
            continue
        values = np.asarray([row[1] for row in rows], dtype=np.float64)
        rv = float(np.dot(values, values))
        downside_values = np.minimum(values, 0.0)
        downside_rv = float(np.dot(downside_values, downside_values))
        if rv <= 0.0 or not math.isfinite(rv) or not math.isfinite(downside_rv):
            continue
        day_ends.append(day_start + SECONDS_PER_DAY)
        realized.append(rv)
        downside.append(downside_rv)
    return day_ends, np.asarray(realized), np.asarray(downside)


def downside_har_variance_forecast(
    daily_rv: Sequence[float],
    daily_downside_rv: Sequence[float],
    *,
    ridge: float = 1.0,
    minimum_training_rows: int = 30,
) -> float:
    """Forecast next-day RV with HAR terms plus downside semivariance state."""
    rv = np.asarray(daily_rv, dtype=np.float64)
    downside = np.asarray(daily_downside_rv, dtype=np.float64)
    if rv.ndim != 1 or downside.ndim != 1 or rv.size != downside.size:
        raise ValueError("daily RV and downside RV must be aligned 1D arrays")
    if rv.size < 22 + minimum_training_rows:
        raise ValueError("not enough complete days for downside-HAR estimation")
    if not np.all(np.isfinite(rv)) or np.any(rv <= 0.0):
        raise ValueError("daily RV must be positive and finite")
    if not np.all(np.isfinite(downside)) or np.any(downside < 0.0):
        raise ValueError("downside RV must be finite and non-negative")
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")

    log_rv = 0.5 * np.log(rv)
    downside_floor = max(float(np.median(rv)) * 1e-10, 1e-18)
    log_downside = 0.5 * np.log(np.maximum(downside, downside_floor))
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in range(22, rv.size):
        rows.append([
            float(log_rv[index - 1]),
            float(np.mean(log_rv[index - 5:index])),
            float(np.mean(log_rv[index - 22:index])),
            float(log_downside[index - 1]),
            float(np.mean(log_downside[index - 5:index])),
        ])
        targets.append(float(log_rv[index]))

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0)
    x_scale = np.where(x_scale > 1e-10, x_scale, 1.0)
    x_standard = (x - x_mean) / x_scale
    y_mean = float(np.mean(y))
    penalty = np.eye(x_standard.shape[1], dtype=np.float64) * ridge
    coefficients = np.linalg.solve(
        x_standard.T @ x_standard + penalty,
        x_standard.T @ (y - y_mean),
    )
    residuals = y - (y_mean + x_standard @ coefficients)
    log_smearing = float(logsumexp(2.0 * residuals) - np.log(residuals.size))

    latest = np.asarray([
        log_rv[-1],
        np.mean(log_rv[-5:]),
        np.mean(log_rv[-22:]),
        log_downside[-1],
        np.mean(log_downside[-5:]),
    ], dtype=np.float64)
    predicted_log_volatility = y_mean + float(
        ((latest - x_mean) / x_scale) @ coefficients
    )
    forecast_log_variance = 2.0 * predicted_log_volatility + log_smearing
    forecast = float(np.exp(np.clip(forecast_log_variance, -30.0, 10.0)))
    if not math.isfinite(forecast) or forecast <= 0.0:
        raise FloatingPointError("downside-HAR produced an invalid variance forecast")
    return forecast


def _qlike(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted variance arrays must be aligned")
    if np.any(actual <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("QLIKE inputs must be positive")
    ratio = actual / predicted
    return ratio - np.log(ratio) - 1.0


def _dm_qlike(
    actual: np.ndarray,
    candidate: np.ndarray,
    benchmark: np.ndarray,
    *,
    hac_lags: int = 5,
) -> dict[str, Any]:
    """Two-sided DM test on QLIKE loss differences with Newey-West HAC."""
    difference = _qlike(actual, candidate) - _qlike(actual, benchmark)
    count = int(difference.size)
    mean_difference = float(np.mean(difference))
    centered = difference - mean_difference
    lag_count = min(max(int(hac_lags), 0), max(count - 1, 0))
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / count) if count else float("nan")
    if not math.isfinite(standard_error) or standard_error <= 1e-15:
        if abs(mean_difference) <= 1e-15:
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = math.copysign(1e12, mean_difference), 0.0
    else:
        statistic = mean_difference / standard_error
        p_value = float(2.0 * student_t.sf(abs(statistic), df=max(count - 1, 1)))
    return {
        "observations": count,
        "mean_qlike_difference_candidate_minus_garch": mean_difference,
        "dm_statistic": float(statistic),
        "p_value": float(p_value),
        "candidate_better_on_mean_loss": mean_difference < 0.0,
        "hac_lags": lag_count,
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min(1.0, (count - rank) * value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    losses = _qlike(actual, predicted)
    errors = predicted - actual
    return {
        "qlike": float(np.mean(losses)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_forecast_to_realized_ratio": float(np.mean(predicted / actual)),
    }


def run_backtest(
    *,
    history_days: int = 125,
    train_window_days: int = 90,
    test_days: int = 30,
    simulations: int = 2_000,
    page_delay_seconds: float = 0.15,
) -> dict[str, Any]:
    if history_days < train_window_days + test_days + 2:
        raise ValueError("history_days must exceed train_window_days + test_days + 1")
    if train_window_days < 60:
        raise ValueError("train_window_days must be at least 60 for HAR-RV")
    if test_days < 20:
        raise ValueError("test_days must be at least 20 for an initial comparison")
    if simulations < 1_000:
        raise ValueError("simulations must be at least 1000")

    now = int(datetime.now(timezone.utc).timestamp())
    end = now - now % 3_600
    start = end - history_days * SECONDS_PER_DAY
    batch = CoinbaseExchangeClient(page_delay_seconds=page_delay_seconds).fetch_range(
        "ETH-USD", "1h", start, end, as_of_timestamp=now
    )
    audit = audit_candle_series(batch.candles)
    if not audit.structurally_valid:
        raise RuntimeError("Coinbase hourly series failed structural validation")
    candles = longest_contiguous_finalized_segment(batch.candles)
    if len(candles) < (train_window_days + test_days) * HOURS_PER_DAY:
        raise RuntimeError("Coinbase returned insufficient contiguous finalized hourly candles")

    all_returns = np.asarray(finalized_contiguous_log_returns(candles), dtype=np.float64)
    return_end_times = [candle.event_time for candle in candles[1:]]
    if len(all_returns) != len(return_end_times):
        raise RuntimeError("return series does not align with candle event times")

    eligible_origins: list[int] = []
    for index, candle in enumerate(candles):
        if candle.event_time % SECONDS_PER_DAY != 0:
            continue
        if index + HOURS_PER_DAY >= len(candles):
            continue
        if len(all_returns[index:index + HOURS_PER_DAY]) != HOURS_PER_DAY:
            continue
        if return_end_times[index] != candle.event_time + 3_600:
            continue
        if return_end_times[index + HOURS_PER_DAY - 1] != candle.event_time + SECONDS_PER_DAY:
            continue
        eligible_origins.append(index)
    origins = eligible_origins[-test_days:]
    if len(origins) < 20:
        raise RuntimeError(f"only {len(origins)} complete daily forecast origins are available")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for origin_index in origins:
        origin = candles[origin_index]
        origin_time = origin.event_time
        history_returns = all_returns[:origin_index]
        history_end_times = return_end_times[:origin_index]
        future_returns = all_returns[origin_index:origin_index + HOURS_PER_DAY]
        future_end_times = return_end_times[origin_index:origin_index + HOURS_PER_DAY]
        if len(future_returns) != HOURS_PER_DAY:
            continue
        expected_future_times = list(range(
            origin_time + 3_600, origin_time + SECONDS_PER_DAY + 1, 3_600
        ))
        if future_end_times != expected_future_times:
            skipped.append({
                "origin": datetime.fromtimestamp(origin_time, timezone.utc).isoformat(),
                "reason": "future target contains a missing or misaligned hour",
            })
            continue

        _, daily_rv, daily_downside = _daily_realized_series(
            history_returns, history_end_times
        )
        if daily_rv.size < train_window_days:
            skipped.append({
                "origin": datetime.fromtimestamp(origin_time, timezone.utc).isoformat(),
                "reason": f"only {daily_rv.size} complete historical days",
            })
            continue

        train_hours = train_window_days * HOURS_PER_DAY
        train_returns = history_returns[-train_hours:]
        train_rv = daily_rv[-train_window_days:]
        train_downside = daily_downside[-train_window_days:]
        realized_variance = float(np.dot(future_returns, future_returns))
        realized_close = float(candles[origin_index + HOURS_PER_DAY].close)
        origin_seed = int(origin_time // SECONDS_PER_DAY) % (2**32 - 1)

        forecasts: dict[str, float] = {}
        model_intervals: dict[str, dict[str, float]] = {}
        try:
            # Score expected variance only for distributions with finite moments
            # under the selected recursion. EGARCH uses Gaussian innovations for
            # this variance comparison; Student-t EGARCH remains a separate
            # interval-only model because its exponential variance recursion has
            # no finite expected variance.
            fit_cache: dict[tuple[str, str], Any] = {}
            for model, distribution, _ in VARIANCE_CONFIG + INTERVAL_CONFIG:
                fit_key = (model, distribution)
                if fit_key not in fit_cache:
                    fit_cache[fit_key] = fit_volatility_model(
                        train_returns, model=model, distribution=distribution,
                    )

            for model, distribution, forecast_name in VARIANCE_CONFIG:
                fitted = fit_cache[(model, distribution)]
                horizon = forecast_price_interval_horizon(
                    fitted,
                    spot_price=float(origin.close),
                    horizon_steps=HOURS_PER_DAY,
                    confidence=0.95,
                    simulations=simulations,
                    random_seed=origin_seed,
                    periods_per_year=365 * HOURS_PER_DAY,
                )
                expected_variance = horizon.expected_cumulative_variance
                if expected_variance is None or not math.isfinite(expected_variance) or expected_variance <= 0.0:
                    raise FloatingPointError(
                        f"{forecast_name} has no finite expected 24-hour variance"
                    )
                forecasts[forecast_name] = expected_variance

            for model, distribution, interval_name in INTERVAL_CONFIG:
                fitted = fit_cache[(model, distribution)]
                horizon = forecast_price_interval_horizon(
                    fitted,
                    spot_price=float(origin.close),
                    horizon_steps=HOURS_PER_DAY,
                    confidence=0.95,
                    simulations=simulations,
                    random_seed=origin_seed,
                    periods_per_year=365 * HOURS_PER_DAY,
                )
                model_intervals[interval_name] = {
                    "lower": horizon.lower_price,
                    "upper": horizon.upper_price,
                    "realized_close": realized_close,
                    "covered": float(horizon.lower_price <= realized_close <= horizon.upper_price),
                    "width_pct_of_spot": float(
                        (horizon.upper_price - horizon.lower_price) / float(origin.close) * 100.0
                    ),
                }

            forecasts["HAR-RV"] = har_rv_forecast(
                train_rv, minimum_training_rows=20
            ).forecast_variance
            forecasts["HAR-RV+downside"] = downside_har_variance_forecast(
                train_rv, train_downside
            )
            forecasts["EWMA-hourly"] = (
                ewma_variance_forecast(train_returns, decay=0.94, initial_window=24)
                * HOURS_PER_DAY
            )
            forecasts["Rolling-RV-22d"] = float(np.mean(train_rv[-22:]))

            if set(forecasts) != set(MODEL_NAMES):
                raise RuntimeError("one or more forecast candidates are missing")
            if not all(math.isfinite(value) and value > 0.0 for value in forecasts.values()):
                raise FloatingPointError("one or more candidates produced invalid variance")
        except Exception as exc:
            skipped.append({
                "origin": datetime.fromtimestamp(origin_time, timezone.utc).isoformat(),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        rows.append({
            "origin": datetime.fromtimestamp(origin_time, timezone.utc).isoformat(),
            "target_end": datetime.fromtimestamp(
                origin_time + SECONDS_PER_DAY, timezone.utc
            ).isoformat(),
            "spot_price": float(origin.close),
            "realized_close": realized_close,
            "realized_24h_variance": realized_variance,
            "forecasts": forecasts,
            "garch_family_intervals": model_intervals,
            "complete_history_days": int(daily_rv.size),
            "training_hourly_returns": int(train_returns.size),
        })

    if len(rows) < 20:
        raise RuntimeError(
            f"only {len(rows)} complete origins survived; at least 20 are required. "
            f"Skipped: {skipped[:5]}"
        )

    actual = np.asarray([row["realized_24h_variance"] for row in rows], dtype=np.float64)
    scorecard: dict[str, Any] = {}
    forecast_arrays: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        predicted = np.asarray([row["forecasts"][name] for row in rows], dtype=np.float64)
        forecast_arrays[name] = predicted
        scorecard[name] = _metrics(actual, predicted)
        scorecard[name]["rank_by_qlike"] = 0

    ranked = sorted(MODEL_NAMES, key=lambda name: scorecard[name]["qlike"])
    for rank, name in enumerate(ranked, start=1):
        scorecard[name]["rank_by_qlike"] = rank

    dm_results: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for name in MODEL_NAMES:
        if name == "GARCH":
            continue
        result = _dm_qlike(actual, forecast_arrays[name], forecast_arrays["GARCH"], hac_lags=5)
        dm_results[name] = result
        raw_p_values[name] = result["p_value"]
    adjusted = _holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        dm_results[name]["holm_adjusted_p_value"] = value

    pairwise_dm_results: dict[str, Any] = {}
    pairwise_raw_p_values: dict[str, float] = {}
    for index, model_a in enumerate(MODEL_NAMES):
        for model_b in MODEL_NAMES[index + 1:]:
            key = f"{model_a} vs {model_b}"
            comparison = _dm_qlike(
                actual, forecast_arrays[model_a], forecast_arrays[model_b], hac_lags=5
            )
            pairwise_dm_results[key] = {
                "model_a": model_a,
                "model_b": model_b,
                "mean_qlike_difference_a_minus_b": comparison[
                    "mean_qlike_difference_candidate_minus_garch"
                ],
                "dm_statistic": comparison["dm_statistic"],
                "p_value": comparison["p_value"],
                "observations": comparison["observations"],
                "hac_lags": comparison["hac_lags"],
            }
            pairwise_raw_p_values[key] = comparison["p_value"]
    pairwise_adjusted = _holm_adjust(pairwise_raw_p_values)
    for key, value in pairwise_adjusted.items():
        pairwise_dm_results[key]["holm_adjusted_p_value"] = value

    interval_coverage: dict[str, Any] = {}
    for model in INTERVAL_NAMES:
        intervals = [row["garch_family_intervals"][model] for row in rows]
        interval_coverage[model] = {
            "nominal": 0.95,
            "empirical_coverage": float(np.mean([item["covered"] for item in intervals])),
            "average_width_pct_of_spot": float(np.mean([
                item["width_pct_of_spot"] for item in intervals
            ])),
            "observations": len(intervals),
        }

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "experiment": "eth_usd_rolling_24h_realized_variance",
        "generated_at": generated_at,
        "data": {
            "provider": batch.provider,
            "instrument": batch.returned_instrument,
            "quote_currency": batch.quote_currency,
            "interval": batch.interval,
            "requested_start": start,
            "requested_end": end,
            "retrieved_at": batch.retrieved_at,
            "raw_response_sha256": batch.raw_response_sha256,
            "finalized_candles_used": len(candles),
            "series_audit": {
                "structurally_valid": audit.structurally_valid,
                "duplicate_event_times": list(audit.duplicate_event_times),
                "out_of_order_indices": list(audit.out_of_order_indices),
                "unobserved_intervals": [list(gap) for gap in audit.unobserved_intervals],
            },
        },
        "design": {
            "forecast_horizon_hours": 24,
            "origin_frequency": "daily UTC boundary; targets do not overlap",
            "training_window_days": train_window_days,
            "requested_test_origins": test_days,
            "completed_test_origins": len(rows),
            "simulations_per_arch_model_origin": simulations,
            "egarch_variance_distribution": "normal; Student-t expected variance is undefined for exponential EGARCH recursion",
            "egarch_interval_distribution": "student_t; interval-only, no expected-variance score",
            "variance_target": "sum of squared finalized hourly log returns over the following 24 hours",
            "common_origin_rule": "all seven candidates must succeed for an origin to be scored",
            "dm_benchmark": "GARCH QLIKE",
            "dm_hac_lags": 5,
            "multiple_testing_adjustment": "Holm-Bonferroni across six GARCH-benchmark comparisons and all 21 candidate pairs",
        },
        "scorecard": scorecard,
        "dm_tests_vs_garch": dm_results,
        "pairwise_dm_tests": pairwise_dm_results,
        "arch_interval_coverage": interval_coverage,
        "forecast_origins": rows,
        "skipped_origins": skipped,
        "interpretation": [
            "This is a preliminary single-venue rolling out-of-sample screen, not a final model selection.",
            "QLIKE, RMSE and MAE score daily realized variance; lower is better.",
            "The downside-HAR challenger uses only prior daily realized variance and downside semivariance.",
            "EGARCH is scored with Gaussian innovations because expected variance is undefined under Student-t innovations in the exponential EGARCH recursion.",
            "Student-t EGARCH is retained as an interval-only candidate; its Monte Carlo sample mean is not treated as an expected variance.",
            "GARCH-family intervals are model-conditional and their empirical coverage is not calibrated by this experiment.",
            "The daily origins have non-overlapping 24-hour targets; the DM tests use a five-day Newey-West HAC estimator.",
            "All 21 pairwise candidate comparisons are reported with Holm-Bonferroni adjusted p-values; this controls multiple testing but does not replace a longer independent holdout or model confidence set.",
            "A model should not be promoted from this short sample without a longer holdout, regime analysis and repeatable results.",
        ],
    }


def _markdown_report(result: dict[str, Any]) -> str:
    data = result["data"]
    design = result["design"]
    lines = [
        "# ETH/USD rolling 24-hour volatility backtest",
        "",
        f"**Generated:** {result['generated_at']}",
        f"**Data:** {data['provider']} {data['instrument']} {data['interval']} candles "
        f"({data['finalized_candles_used']} finalized candles)",
        f"**OOS origins:** {design['completed_test_origins']} daily origins, "
        f"{design['training_window_days']}-day rolling training window",
        "",
        "## Common-origin scorecard",
        "",
        "| Rank | Candidate | QLIKE ↓ | RMSE ↓ | MAE ↓ | Forecast / realized ratio |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    ranked = sorted(
        result["scorecard"].items(),
        key=lambda item: item[1]["rank_by_qlike"],
    )
    for name, metrics in ranked:
        lines.append(
            f"| {metrics['rank_by_qlike']} | {name} | {metrics['qlike']:.6f} | "
            f"{metrics['rmse']:.8f} | {metrics['mae']:.8f} | "
            f"{metrics['mean_forecast_to_realized_ratio']:.3f} |"
        )
    lines.extend([
        "",
        "## Candidate comparisons against GARCH",
        "",
        "| Candidate | Mean QLIKE difference vs GARCH | DM statistic | p-value | Holm adjusted p |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, values in result["dm_tests_vs_garch"].items():
        lines.append(
            f"| {name} | {values['mean_qlike_difference_candidate_minus_garch']:.6f} | "
            f"{values['dm_statistic']:.3f} | {values['p_value']:.4f} | "
            f"{values['holm_adjusted_p_value']:.4f} |"
        )
    lines.extend([
        "",
        "## All-pairs comparisons (Holm adjusted)",
        "",
        "| Model A | Model B | Mean QLIKE difference A − B | DM statistic | p-value | Holm adjusted p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for values in result["pairwise_dm_tests"].values():
        lines.append(
            f"| {values['model_a']} | {values['model_b']} | "
            f"{values['mean_qlike_difference_a_minus_b']:.6f} | "
            f"{values['dm_statistic']:.3f} | {values['p_value']:.4f} | "
            f"{values['holm_adjusted_p_value']:.4f} |"
        )
    lines.extend([
        "",
        "Negative A − B means model A had lower average QLIKE. Pairwise p-values are Holm-adjusted "
        "across all 21 comparisons; this is still an exploratory 30-origin, single-venue test.",
        "",
    ])
    lines.extend([
        "",
        "Negative mean QLIKE difference means the candidate had lower average loss than GARCH. "
        "The p-values are exploratory because the holdout is short and only one venue is tested.",
        "",
        "## GARCH-family 95% interval coverage",
        "",
        "| Model | Nominal | Empirical coverage | Average interval width (% of spot) |",
        "|---|---:|---:|---:|",
    ])
    for name, values in result["arch_interval_coverage"].items():
        lines.append(
            f"| {name} | {values['nominal']:.0%} | {values['empirical_coverage']:.1%} | "
            f"{values['average_width_pct_of_spot']:.2f}% |"
        )
    lines.extend([
        "",
        "## Interpretation and limits",
        "",
        *[f"- {item}" for item in result["interpretation"]],
        f"- Skipped origins: {len(result['skipped_origins'])}.",
        f"- Source response SHA-256: {data['raw_response_sha256']}.",
        "",
        "The comparison does not establish a winning model. Any promotion requires a longer "
        "holdout, more than one venue, regime-specific analysis and interval calibration.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-days", type=int, default=125)
    parser.add_argument("--train-window-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/research/eth-volatility-backtest"
    ))
    args = parser.parse_args()

    result = run_backtest(
        history_days=args.history_days,
        train_window_days=args.train_window_days,
        test_days=args.test_days,
        simulations=args.simulations,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "latest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "latest.md").write_text(
        _markdown_report(result), encoding="utf-8"
    )
    print(json.dumps({
        "status": "succeeded",
        "generated_at": result["generated_at"],
        "completed_test_origins": result["design"]["completed_test_origins"],
        "best_qlike_candidate": min(
            result["scorecard"], key=lambda name: result["scorecard"][name]["qlike"]
        ),
        "output_dir": str(args.output_dir),
        "skipped_origins": len(result["skipped_origins"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
