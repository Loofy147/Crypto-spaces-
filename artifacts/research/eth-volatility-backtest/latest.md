# ETH/USD rolling 24-hour volatility backtest

**Generated:** 2026-11-07T01:31:34.796785Z
**Data:** coinbase ETH-USD 1h candles (3000 finalized candles)
**OOS origins:** 30 daily origins, 90-day rolling training window

## Common-origin scorecard

| Rank | Candidate | QLIKE ↓ | RMSE ↓ | MAE ↓ | Forecast / realized ratio |
|---:|---|---:|---:|---:|---:|
| 1 | HAR-RV+downside | 0.367117 | 0.00059082 | 0.00040619 | 1.834 |
| 2 | HAR-RV | 0.380140 | 0.00059849 | 0.00044754 | 2.260 |
| 3 | EGARCH-normal | 0.465613 | 0.00068903 | 0.00061052 | 3.143 |
| 4 | Rolling-RV-22d | 0.481460 | 0.00064791 | 0.00044872 | 2.041 |
| 5 | EWMA-hourly | 0.701552 | 0.00085897 | 0.00060191 | 2.906 |
| 6 | GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |
| 7 | GJR-GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |

## Candidate comparisons against GARCH

| Candidate | Mean QLIKE difference vs GARCH | DM statistic | p-value | Holm adjusted p |
|---|---:|---:|---:|---:|
| EGARCH-normal | -0.255243 | -4.550 | 0.0001 | 0.0004 |
| GJR-GARCH | 0.000000 | 0.188 | 0.8521 | 1.0000 |
| HAR-RV | -0.340716 | -5.274 | 0.0000 | 0.0001 |
| HAR-RV+downside | -0.353739 | -4.856 | 0.0000 | 0.0002 |
| EWMA-hourly | -0.019304 | -0.149 | 0.8827 | 1.0000 |
| Rolling-RV-22d | -0.239396 | -2.137 | 0.0412 | 0.1236 |

## All-pairs comparisons (Holm adjusted)

| Model A | Model B | Mean QLIKE difference A − B | DM statistic | p-value | Holm adjusted p |
|---|---|---:|---:|---:|---:|
| GARCH | EGARCH-normal | 0.255243 | 4.550 | 0.0001 | 0.0015 |
| GARCH | GJR-GARCH | -0.000000 | -0.188 | 0.8521 | 1.0000 |
| GARCH | HAR-RV | 0.340716 | 5.274 | 0.0000 | 0.0002 |
| GARCH | HAR-RV+downside | 0.353739 | 4.856 | 0.0000 | 0.0007 |
| GARCH | EWMA-hourly | 0.019304 | 0.149 | 0.8827 | 1.0000 |
| GARCH | Rolling-RV-22d | 0.239396 | 2.137 | 0.0412 | 0.4119 |
| EGARCH-normal | GJR-GARCH | -0.255243 | -4.550 | 0.0001 | 0.0015 |
| EGARCH-normal | HAR-RV | 0.085473 | 2.943 | 0.0063 | 0.0761 |
| EGARCH-normal | HAR-RV+downside | 0.098496 | 2.058 | 0.0487 | 0.4119 |
| EGARCH-normal | EWMA-hourly | -0.235939 | -2.219 | 0.0345 | 0.3790 |
| EGARCH-normal | Rolling-RV-22d | -0.015848 | -0.226 | 0.8226 | 1.0000 |
| GJR-GARCH | HAR-RV | 0.340716 | 5.274 | 0.0000 | 0.0002 |
| GJR-GARCH | HAR-RV+downside | 0.353739 | 4.856 | 0.0000 | 0.0007 |
| GJR-GARCH | EWMA-hourly | 0.019304 | 0.149 | 0.8827 | 1.0000 |
| GJR-GARCH | Rolling-RV-22d | 0.239396 | 2.137 | 0.0412 | 0.4119 |
| HAR-RV | HAR-RV+downside | 0.013023 | 0.383 | 0.7044 | 1.0000 |
| HAR-RV | EWMA-hourly | -0.321412 | -3.232 | 0.0031 | 0.0428 |
| HAR-RV | Rolling-RV-22d | -0.101320 | -1.440 | 0.1605 | 0.9632 |
| HAR-RV+downside | EWMA-hourly | -0.334435 | -3.468 | 0.0017 | 0.0249 |
| HAR-RV+downside | Rolling-RV-22d | -0.114343 | -1.569 | 0.1275 | 0.8927 |
| EWMA-hourly | Rolling-RV-22d | 0.220092 | 3.098 | 0.0043 | 0.0559 |

Negative A − B means model A had lower average QLIKE. Pairwise p-values are Holm-adjusted across all 21 comparisons; this is still an exploratory 30-origin, single-venue test.


Negative mean QLIKE difference means the candidate had lower average loss than GARCH. The p-values are exploratory because the holdout is short and only one venue is tested.

## GARCH-family 95% interval coverage

| Model | Nominal | Empirical coverage | Average interval width (% of spot) |
|---|---:|---:|---:|
| GARCH | 95% | 90.0% | 10.68% |
| EGARCH-student_t | 95% | 90.0% | 11.14% |
| GJR-GARCH | 95% | 90.0% | 10.68% |

## Interpretation and limits

- This is a preliminary single-venue rolling out-of-sample screen, not a final model selection.
- QLIKE, RMSE and MAE score daily realized variance; lower is better.
- The downside-HAR challenger uses only prior daily realized variance and downside semivariance.
- EGARCH is scored with Gaussian innovations because expected variance is undefined under Student-t innovations in the exponential EGARCH recursion.
- Student-t EGARCH is retained as an interval-only candidate; its Monte Carlo sample mean is not treated as an expected variance.
- GARCH-family intervals are model-conditional and their empirical coverage is not calibrated by this experiment.
- The daily origins have non-overlapping 24-hour targets; the DM tests use a five-day Newey-West HAC estimator.
- All 21 pairwise candidate comparisons are reported with Holm-Bonferroni adjusted p-values; this controls multiple testing but does not replace a longer independent holdout or model confidence set.
- A model should not be promoted from this short sample without a longer holdout, regime analysis and repeatable results.
- Skipped origins: 0.
- Source response SHA-256: 248b4282dd7d1a8aadada1c0e3d01fbf9098f3d30aca7940e469fa786005810f.

The comparison does not establish a winning model. Any promotion requires a longer holdout, more than one venue, regime-specific analysis and interval calibration.
