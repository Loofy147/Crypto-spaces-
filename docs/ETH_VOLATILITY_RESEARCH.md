# ETH/USD Volatility Research

## Purpose

This work extends the existing Python quantitative engine with a reproducible
conditional-volatility research layer. It is initially scoped to ETH/USD, but
the model functions are asset-agnostic.

The module estimates conditional variance and converts a one-period forecast
into a probability-based price interval. It does not forecast directional
returns, recommend trades, or grant execution permission.

## Implemented in the first research increment

Entry point: \`src/models/volatility_forecasting.py\`

| Method | Current implementation | Role |
|---|---|---|
| GARCH(1,1) | Gaussian or standardized Student-t innovations | Baseline volatility clustering |
| EGARCH(1,1) | Gaussian or standardized Student-t innovations | Asymmetric response to shocks |
| GJR-GARCH(1,1) | Gaussian or standardized Student-t innovations | Threshold/downside variance response |
| EWMA | Configurable decay, default 0.94 | Responsive benchmark |
| HAR-RV | Daily, weekly and monthly log-RV regressors | Multi-horizon realized-volatility benchmark |
| Realized variance | Sum of squared intraday log returns | Ex-post variation proxy |
| Bipower variation | Intraday bipower estimator | Continuous-path variation proxy |
| Jump variation | Non-negative RV minus BPV proxy | Jump contribution diagnostic |

The GARCH-family estimators use constrained maximum likelihood via SciPy.
The Student-t distribution is standardized to unit variance; its degrees of
freedom must therefore exceed two. Returns are accepted as decimal log returns
and internally scaled to percentage points for estimation.

The price interval function maps the next-period variance estimate to a
conditional return distribution. The resulting price interval is a model-based
quantile range, not a support/resistance estimate and not a guaranteed bound.

## Minimal usage

\`\`\`python
from src.models.volatility_forecasting import (
    fit_volatility_model,
    forecast_price_interval,
)

# Completed, UTC-aligned daily log returns, expressed as decimals.
fit = fit_volatility_model(
    daily_log_returns,
    model="GARCH",
    distribution="student_t",
)

forecast = forecast_price_interval(
    fit,
    spot_price=2300.0,
    confidence=0.95,
    periods_per_year=365,
)

print(forecast.conditional_volatility)
print(forecast.annualized_volatility)
print(forecast.lower_price, forecast.median_price, forecast.upper_price)
\`\`\`

For other sampling frequencies, use one fixed-frequency return series and
annualize consistently:

- Daily: 365 periods/year.
- Four-hour: 365 × 6 periods/year.
- Hourly: 365 × 24 periods/year.

Do not mix bars from different exchanges, quote currencies, time zones or
candle boundaries in a single return series.

## Interpretation and limitations

1. GARCH estimates conditional variance, not the sign of the next return.
2. A Student-t interval is heavier-tailed than a Gaussian interval when its
   degrees of freedom are finite. Estimated degrees of freedom are
   sample- and window-dependent.
3. GJR-GARCH and EGARCH test different asymmetry mechanisms; asymmetry should
   be assessed empirically rather than assumed.
4. HAR-RV requires positive daily realized-variance observations and enough
   history to form daily, weekly and monthly regressors.
5. Daily squared returns are a noisy realized-variance proxy. Intraday realized
   variance is preferred for out-of-sample variance scoring.
6. Variance combination requires explicit non-negative weights that sum to
   one. It intentionally does not learn weights from in-sample fit.
7. The first increment does not yet implement APARCH, FIGARCH, regime-switching
   GARCH, rolling-origin backtests, exchange ingestion, database persistence,
   a live API, a scheduler, or a dashboard.

## Required data contract

Each observation must identify:

- \`venue\`: exchange/provider identifier.
- \`instrument\`: exact symbol (e.g. \`ETH-USD\` or \`ETH-USDT\`).
- \`quote_currency\`: USD and USDT must remain distinct.
- \`interval\`: fixed bar interval (\`1h\`, \`4h\`, \`1d\`, etc.).
- \`event_time\`: canonical UTC bar-end timestamp.
- \`open\`, \`high\`, \`low\`, \`close\`, \`volume\`: source values.
- \`source_timestamp\`: original exchange timestamp and documented meaning.
- \`ingested_at\`: UTC ingestion timestamp.
- \`is_final\`: whether the candle is complete.
- \`source_payload_hash\`: provenance/integrity reference for raw observation.

Validation must reject duplicate event keys, non-monotonic timestamps, invalid
OHLC relationships, negative volume, non-finite prices, missing intervals,
partial candles in completed-bar training sets, and unexplained cross-source
divergence. Keep raw data immutable; store normalized observations as derived
artifacts with transformation version and source provenance.

## Validation protocol before publishing a live forecast

1. Freeze a canonical data window and record its source, instrument, quote
   currency, interval, timezone, retrieval time and content hash.
2. Fit each candidate only on observations available at the forecast origin.
3. Use rolling or expanding time-ordered evaluation; never randomly shuffle
   time-series data.
4. Compare against EWMA and a simple historical-variance benchmark.
5. Score variance forecasts with QLIKE, RMSE and MAE. Prefer intraday RV as the
   realized-variance proxy where available.
6. Assess predictive intervals using empirical coverage, Kupiec unconditional
   coverage and Christoffersen independence tests.
7. Compare candidate models with paired out-of-sample loss differentials;
   account for overlapping multi-period targets.
8. Report forecast errors by market regime and time horizon, not just one
   aggregate metric.
9. Version model specification, estimation window, fitted parameters, forecast
   origin, input-data hash and scoring result.

A model is not promoted merely because its in-sample likelihood or AIC is best.
The live candidate must demonstrate stable out-of-sample calibration and
incremental value over the benchmark.

## Next gates

| Gate | Work | Exit evidence |
|---|---|---|
| V01 | Canonical ETH spot candles from two independent venues | Provenance, temporal and OHLC integrity reports |
| V02 | Rolling-origin baseline evaluation | GARCH / EGARCH / GJR / EWMA scorecard |
| V03 | Intraday realized-volatility pipeline | 1h, 4h and daily RV/BPV/jump measures |
| V04 | HAR-RV and distributional comparisons | OOS loss and interval-calibration report |
| V05 | APARCH, FIGARCH and regime-switching candidates | Incremental OOS value demonstrated |
| V06 | Forecast ensemble | Weights learned only from past OOS loss; stability report |
| V07 | Persistence, API and scheduling | Replayable forecast records and monitoring |
| V08 | Dashboard | UI reads persisted, versioned forecast artifacts |

## Current status

The first model module and deterministic unit tests are implemented in the
repository. This does not establish live-market connectivity, independently
validated ETH/USD forecast accuracy, scheduled execution, or deployment. Those
remain separate acceptance gates.
