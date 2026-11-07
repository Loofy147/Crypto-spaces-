# ETH/USD rolling 24-hour volatility backtest

**Generated:** 2026-11-07T01:20:43.216506Z
**Data:** coinbase ETH-USD 1h candles (3000 finalized candles)
**OOS origins:** 30 daily origins, 90-day rolling training window

## Common-origin scorecard

| Rank | Candidate | QLIKE ↓ | RMSE ↓ | MAE ↓ | Forecast / realized ratio |
|---:|---|---:|---:|---:|---:|
| 1 | EGARCH-normal | 0.465613 | 0.00068903 | 0.00061052 | 3.143 |
| 2 | Rolling-RV-22d | 0.481460 | 0.00064791 | 0.00044872 | 2.041 |
| 3 | HAR-RV | 0.500931 | 0.00064692 | 0.00041561 | 1.440 |
| 4 | HAR-RV+downside | 0.582922 | 0.00066514 | 0.00042223 | 1.209 |
| 5 | EWMA-hourly | 0.701552 | 0.00085897 | 0.00060191 | 2.906 |
| 6 | GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |
| 7 | GJR-GARCH | 0.720856 | 0.00142398 | 0.00115079 | 5.613 |

## Candidate comparisons against GARCH

| Candidate | Mean QLIKE difference vs GARCH | DM statistic | p-value | Holm adjusted p |
|---|---:|---:|---:|---:|
| EGARCH-normal | -0.255243 | -4.550 | 0.0001 | 0.0005 |
| GJR-GARCH | 0.000000 | 0.188 | 0.8521 | 1.0000 |
| HAR-RV | -0.219925 | -2.108 | 0.0438 | 0.2060 |
| HAR-RV+downside | -0.137934 | -1.159 | 0.2558 | 0.7675 |
| EWMA-hourly | -0.019304 | -0.149 | 0.8827 | 1.0000 |
| Rolling-RV-22d | -0.239396 | -2.137 | 0.0412 | 0.2060 |

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
- The daily origins have non-overlapping 24-hour targets; the DM tests still use a five-day Newey-West HAC estimator.
- A model should not be promoted from this short sample without a longer holdout, regime analysis and repeatable results.
- Skipped origins: 0.
- Source response SHA-256: 48598e6be5eee8a1872754e0692e496a89de17f60b628c3959c1f06b7f5b57f3.

The comparison does not establish a winning model. Any promotion requires a longer holdout, more than one venue, regime-specific analysis and interval calibration.
