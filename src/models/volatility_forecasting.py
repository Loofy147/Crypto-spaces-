"""Conditional volatility forecasting primitives for digital-asset returns.

Returns are decimal log returns at one fixed sampling frequency. ARCH-family
estimation uses percentage-point returns internally for numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import norm, t

VolatilityModel = Literal["GARCH", "EGARCH", "GJR-GARCH"]
InnovationDistribution = Literal["normal", "student_t"]


@dataclass(frozen=True)
class VolatilityModelFit:
    model: VolatilityModel
    distribution: InnovationDistribution
    parameters: Mapping[str, float]
    observations: int
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    conditional_variances_pct2: tuple[float, ...]
    standardized_residuals: tuple[float, ...]


@dataclass(frozen=True)
class PriceInterval:
    spot_price: float
    confidence: float
    lower_price: float
    median_price: float
    upper_price: float
    conditional_volatility: float
    annualized_volatility: float
    mean_log_return: float
    degrees_of_freedom: float | None


@dataclass(frozen=True)
class VarianceForecastScore:
    observations: int
    qlike: float
    rmse: float
    mae: float


@dataclass(frozen=True)
class HARForecast:
    forecast_volatility: float
    forecast_variance: float
    coefficients: tuple[float, float, float, float]
    observations: int
    residual_rmse: float


def _validated_returns(returns: Sequence[float], minimum: int = 40) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("returns must be a one-dimensional sequence")
    if values.size < minimum:
        raise ValueError(f"at least {minimum} return observations are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must contain only finite values")
    if float(np.var(values)) <= 1e-14:
        raise ValueError("returns have insufficient variation to estimate volatility")
    return values


def _expected_abs_standardized_t(nu: float) -> float:
    if nu <= 2.0:
        raise ValueError("Student-t degrees of freedom must exceed 2")
    log_value = (
        np.log(2.0) + 0.5 * np.log(nu - 2.0)
        + gammaln((nu + 1.0) / 2.0) - 0.5 * np.log(np.pi)
        - np.log(nu - 1.0) - gammaln(nu / 2.0)
    )
    return float(np.exp(log_value))


def _unpack(
    vector: np.ndarray,
    model: VolatilityModel,
    distribution: InnovationDistribution,
) -> dict[str, float]:
    p: dict[str, float] = {"mu": float(vector[0])}
    if model == "GARCH":
        p.update(omega=float(np.exp(vector[1])), alpha=float(vector[2]), beta=float(vector[3]))
        nu_index = 4
    elif model == "GJR-GARCH":
        p.update(
            omega=float(np.exp(vector[1])), alpha=float(vector[2]),
            gamma=float(vector[3]), beta=float(vector[4]),
        )
        nu_index = 5
    else:
        p.update(
            omega=float(vector[1]), alpha=float(vector[2]),
            gamma=float(vector[3]), beta=float(vector[4]),
        )
        nu_index = 5
    if distribution == "student_t":
        p["nu"] = float(vector[nu_index])
    return p


def _filter_variances(
    returns_pct: np.ndarray, p: Mapping[str, float], model: VolatilityModel
) -> tuple[np.ndarray, np.ndarray]:
    n = returns_pct.size
    h = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)
    residuals = returns_pct - p["mu"]
    h[0] = max(float(np.var(returns_pct, ddof=1)), 1e-8)
    z[0] = residuals[0] / np.sqrt(h[0])
    for i in range(1, n):
        hp = max(float(h[i - 1]), 1e-12)
        ep = float(residuals[i - 1])
        if model == "GARCH":
            hi = p["omega"] + p["alpha"] * ep**2 + p["beta"] * hp
        elif model == "GJR-GARCH":
            downside = ep**2 if ep < 0.0 else 0.0
            hi = p["omega"] + p["alpha"] * ep**2 + p["gamma"] * downside + p["beta"] * hp
        else:
            nu = p.get("nu")
            eabs = _expected_abs_standardized_t(nu) if nu is not None else np.sqrt(2.0 / np.pi)
            zp = ep / np.sqrt(hp)
            log_h = p["omega"] + p["beta"] * np.log(hp) + p["alpha"] * (abs(zp) - eabs) + p["gamma"] * zp
            hi = float(np.exp(np.clip(log_h, -30.0, 30.0)))
        if not np.isfinite(hi) or hi <= 0.0:
            raise FloatingPointError("variance recursion produced an invalid value")
        h[i] = hi
        z[i] = residuals[i] / np.sqrt(hi)
    return h, z


def _nll(
    vector: np.ndarray, returns_pct: np.ndarray,
    model: VolatilityModel, distribution: InnovationDistribution,
) -> float:
    try:
        p = _unpack(vector, model, distribution)
        h, _ = _filter_variances(returns_pct, p, model)
        z = (returns_pct - p["mu"]) / np.sqrt(h)
        log_sigma = 0.5 * np.log(h)
        if distribution == "normal":
            log_density = -0.5 * (np.log(2.0 * np.pi) + z**2) - log_sigma
        else:
            nu = p["nu"]
            log_density = (
                gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
                - 0.5 * np.log(np.pi * (nu - 2.0))
                - ((nu + 1.0) / 2.0) * np.log1p(z**2 / (nu - 2.0))
                - log_sigma
            )
        return float(-np.sum(log_density)) if np.all(np.isfinite(log_density)) else 1e100
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return 1e100


def fit_volatility_model(
    returns: Sequence[float],
    model: VolatilityModel = "GARCH",
    distribution: InnovationDistribution = "student_t",
    max_iterations: int = 2_000,
) -> VolatilityModelFit:
    """Fit GARCH, EGARCH or GJR-GARCH by constrained maximum likelihood."""
    if model not in ("GARCH", "EGARCH", "GJR-GARCH"):
        raise ValueError(f"unsupported volatility model: {model}")
    if distribution not in ("normal", "student_t"):
        raise ValueError(f"unsupported innovation distribution: {distribution}")
    if max_iterations < 100:
        raise ValueError("max_iterations must be at least 100")

    values = _validated_returns(returns) * 100.0
    sample_var = max(float(np.var(values, ddof=1)), 1e-8)
    mu = float(np.mean(values))
    if model == "GARCH":
        x0 = [mu, np.log(sample_var * 0.05), 0.08, 0.88]
        bounds = [(-100, 100), (-20, 10), (1e-6, 0.999), (1e-6, 0.999)]
        constraint = {"type": "ineq", "fun": lambda x: 0.999 - x[2] - x[3]}
    elif model == "GJR-GARCH":
        x0 = [mu, np.log(sample_var * 0.05), 0.06, 0.05, 0.85]
        bounds = [
            (-100, 100), (-20, 10), (1e-6, 0.999),
            (0.0, 0.999), (1e-6, 0.999),
        ]
        constraint = {"type": "ineq", "fun": lambda x: 0.999 - x[2] - 0.5 * x[3] - x[4]}
    else:
        x0 = [mu, 0.1 * np.log(sample_var), 0.12, -0.05, 0.90]
        bounds = [(-100, 100), (-20, 20), (1e-6, 5), (-5, 5), (0.0, 0.999)]
        constraint = None
    if distribution == "student_t":
        x0.append(8.0)
        bounds.append((2.05, 100.0))

    constraints = [constraint] if constraint is not None else []
    result = minimize(
        _nll, np.asarray(x0, dtype=np.float64),
        args=(values, model, distribution), method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": max_iterations, "ftol": 1e-9, "disp": False},
    )
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e90:
        raise RuntimeError(f"{model} maximum-likelihood fit failed: {result.message}")

    p = _unpack(result.x, model, distribution)
    h, z = _filter_variances(values, p, model)
    n = int(values.size)
    k = int(result.x.size)
    ll = float(-result.fun)
    return VolatilityModelFit(
        model=model, distribution=distribution, parameters=p, observations=n,
        log_likelihood=ll, aic=float(2 * k - 2 * ll),
        bic=float(np.log(n) * k - 2 * ll), converged=True,
        conditional_variances_pct2=tuple(float(v) for v in h),
        standardized_residuals=tuple(float(v) for v in z),
    )


def next_period_variance(fit: VolatilityModelFit) -> float:
    """Return one-step conditional variance in decimal-return units."""
    p = fit.parameters
    hp = max(fit.conditional_variances_pct2[-1], 1e-12)
    zp = fit.standardized_residuals[-1]
    ep = zp * np.sqrt(hp)
    if fit.model == "GARCH":
        h_next = p["omega"] + p["alpha"] * ep**2 + p["beta"] * hp
    elif fit.model == "GJR-GARCH":
        downside = ep**2 if ep < 0.0 else 0.0
        h_next = p["omega"] + p["alpha"] * ep**2 + p["gamma"] * downside + p["beta"] * hp
    else:
        nu = p.get("nu")
        eabs = _expected_abs_standardized_t(nu) if nu is not None else np.sqrt(2.0 / np.pi)
        log_h = p["omega"] + p["beta"] * np.log(hp) + p["alpha"] * (abs(zp) - eabs) + p["gamma"] * zp
        h_next = float(np.exp(np.clip(log_h, -30.0, 30.0)))
    if not np.isfinite(h_next) or h_next <= 0.0:
        raise FloatingPointError("one-step forecast variance is invalid")
    return float(h_next / 10_000.0)


def price_interval_from_variance(
    spot_price: float, conditional_variance: float, confidence: float = 0.95,
    mean_log_return: float = 0.0,
    distribution: InnovationDistribution = "student_t",
    degrees_of_freedom: float | None = 8.0,
    periods_per_year: int = 365,
) -> PriceInterval:
    """Convert one-period variance to conditional price quantiles."""
    if not np.isfinite(spot_price) or spot_price <= 0.0:
        raise ValueError("spot_price must be positive and finite")
    if not np.isfinite(conditional_variance) or conditional_variance <= 0.0:
        raise ValueError("conditional_variance must be positive and finite")
    if not 0.5 < confidence < 0.999:
        raise ValueError("confidence must be between 0.5 and 0.999")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    sigma = float(np.sqrt(conditional_variance))
    tail = (1.0 - confidence) / 2.0
    if distribution == "normal":
        q_lo, q_hi, nu_value = float(norm.ppf(tail)), float(norm.ppf(1.0 - tail)), None
    elif distribution == "student_t":
        if degrees_of_freedom is None or degrees_of_freedom <= 2.0:
            raise ValueError("Student-t degrees_of_freedom must exceed 2")
        scale = float(np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom))
        q_lo = float(t.ppf(tail, degrees_of_freedom) * scale)
        q_hi = float(t.ppf(1.0 - tail, degrees_of_freedom) * scale)
        nu_value = float(degrees_of_freedom)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")
    return PriceInterval(
        spot_price=float(spot_price), confidence=float(confidence),
        lower_price=spot_price * float(np.exp(mean_log_return + sigma * q_lo)),
        median_price=spot_price * float(np.exp(mean_log_return)),
        upper_price=spot_price * float(np.exp(mean_log_return + sigma * q_hi)),
        conditional_volatility=sigma,
        annualized_volatility=sigma * float(np.sqrt(periods_per_year)),
        mean_log_return=float(mean_log_return), degrees_of_freedom=nu_value,
    )


def forecast_price_interval(
    fit: VolatilityModelFit, spot_price: float, confidence: float = 0.95,
    periods_per_year: int = 365,
) -> PriceInterval:
    return price_interval_from_variance(
        spot_price, next_period_variance(fit), confidence,
        mean_log_return=fit.parameters["mu"] / 100.0,
        distribution=fit.distribution, degrees_of_freedom=fit.parameters.get("nu"),
        periods_per_year=periods_per_year,
    )


def ewma_variance_forecast(
    returns: Sequence[float], decay: float = 0.94, initial_window: int = 20
) -> float:
    """One-step EWMA variance benchmark on decimal returns."""
    values = _validated_returns(returns, minimum=max(3, initial_window))
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be strictly between 0 and 1")
    if initial_window < 2 or initial_window > values.size:
        raise ValueError("invalid initial_window")
    variance = float(np.var(values[:initial_window], ddof=1))
    for observation in values[initial_window:]:
        variance = decay * variance + (1.0 - decay) * float(observation**2)
    return max(variance, 1e-16)


def realized_variance(intraday_returns: Sequence[float]) -> float:
    values = np.asarray(intraday_returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("intraday_returns must be a non-empty finite 1D sequence")
    return float(np.dot(values, values))


def bipower_variation(intraday_returns: Sequence[float]) -> float:
    values = np.asarray(intraday_returns, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("at least two finite intraday returns are required")
    return float((np.pi / 2.0) * np.sum(np.abs(values[1:]) * np.abs(values[:-1])))


def jump_variation(intraday_returns: Sequence[float]) -> float:
    return max(realized_variance(intraday_returns) - bipower_variation(intraday_returns), 0.0)


def har_rv_forecast(
    daily_realized_variances: Sequence[float], minimum_training_rows: int = 20
) -> HARForecast:
    """Forecast next-day log realized volatility from daily/weekly/monthly terms."""
    rv = np.asarray(daily_realized_variances, dtype=np.float64)
    if rv.ndim != 1 or not np.all(np.isfinite(rv)) or np.any(rv <= 0.0):
        raise ValueError("daily_realized_variances must be positive finite 1D values")
    if minimum_training_rows < 5 or rv.size < 22 + minimum_training_rows:
        raise ValueError(f"at least {22 + max(5, minimum_training_rows)} observations are required")
    log_vol = 0.5 * np.log(rv)
    rows: list[list[float]] = []
    targets: list[float] = []
    for idx in range(22, log_vol.size):
        prior = log_vol[:idx]
        rows.append([1.0, float(prior[-1]), float(np.mean(prior[-5:])), float(np.mean(prior[-22:]))])
        targets.append(float(log_vol[idx]))
    x, y = np.asarray(rows), np.asarray(targets)
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    latest = np.array([1.0, log_vol[-1], np.mean(log_vol[-5:]), np.mean(log_vol[-22:])])
    predicted_log_volatility = float(latest @ coefficients)
    # Duan smearing correction converts the log-volatility centre into an
    # expected variance forecast. The smearing factor uses training residuals
    # only, so no future target information enters the variance forecast.
    log_smearing = float(logsumexp(2.0 * residuals) - np.log(residuals.size))
    forecast_log_variance = predicted_log_volatility * 2.0 + log_smearing
    forecast_variance = float(np.exp(np.clip(forecast_log_variance, -30.0, 10.0)))
    forecast_vol = float(np.sqrt(forecast_variance))
    return HARForecast(
        forecast_volatility=forecast_vol, forecast_variance=forecast_variance,
        coefficients=tuple(float(v) for v in coefficients), observations=int(y.size),
        residual_rmse=float(np.sqrt(np.mean(residuals**2))),
    )


def score_variance_forecasts(
    predicted_variances: Sequence[float], realized_variances: Sequence[float]
) -> VarianceForecastScore:
    """Score aligned OOS forecasts; prefer intraday RV over squared daily returns."""
    predicted = np.asarray(predicted_variances, dtype=np.float64)
    realized = np.asarray(realized_variances, dtype=np.float64)
    if predicted.ndim != 1 or realized.ndim != 1 or predicted.size != realized.size or predicted.size == 0:
        raise ValueError("predicted and realized variances must be aligned non-empty 1D arrays")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(realized)) or np.any(predicted <= 0) or np.any(realized <= 0):
        raise ValueError("variance inputs must be positive and finite")
    errors = predicted - realized
    ratio = realized / predicted
    return VarianceForecastScore(
        observations=int(predicted.size),
        qlike=float(np.mean(ratio - np.log(ratio) - 1.0)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
    )


def combine_variance_forecasts(forecasts: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Combine variance forecasts using explicit non-negative simplex weights."""
    if not forecasts or set(forecasts) != set(weights):
        raise ValueError("forecasts and weights must have the same non-empty keys")
    values = np.asarray(list(forecasts.values()), dtype=np.float64)
    w = np.asarray([weights[key] for key in forecasts], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("forecast variances must be positive and finite")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isclose(np.sum(w), 1.0, atol=1e-8):
        raise ValueError("weights must be finite, non-negative and sum to one")
    return float(np.dot(values, w))
