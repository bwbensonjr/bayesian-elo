# Expanding-Window Sequential Comparison: Classical Elo vs Bayesian Elo

## Methodology

- **Data:** 29 Premier League seasons (1994-95 through 2021-22 used as test seasons)
- **Evaluation matches:** 10651 (all matches from seasons 2 through 29)
- **Classical Elo:** Run sequentially over all seasons; each prediction is made before observing that match's result (online updates). Parameters: K=3, home_field=20, spread_factor=55.
- **Bayesian Elo:** Expanding-window retraining. For each test season s, a full MCMC fit is run on all prior seasons (1..s-1). Predictions for season s use the final-period posterior ratings from that fit. This gives the Bayesian model access to all historical data while ensuring genuinely forward-looking predictions.
- **Spread calibration:** Computed per-window from training data only. Overall calibration: E[spread|H]=1.88, E[spread|D]=0.00, E[spread|A]=-1.71.

## Outcome Prediction Accuracy

- **Bayesian categorical accuracy (H/D/A):** 51.6%
- **Bayesian log-loss:** 0.9985
- **Win/loss accuracy (excl. draws):** Classical Elo 70.2%, Bayesian 69.3%

## Point Spread Prediction

| Metric | Classical Elo | Bayesian Elo |
|--------|--------------|-------------|
| MAE    | 1.2478 | 1.2646 |
| RMSE   | 1.6173 | 1.6418 |

## Summary Comparison

| Metric | Classical Elo | Bayesian Elo |
|--------|--------------|-------------|
| Win accuracy (excl. draws) | 70.2% | 69.3% |
| Categorical accuracy (H/D/A) | N/A | 51.6% |
| Log-loss | N/A | 0.9985 |
| Spread MAE | 1.2478 | 1.2646 |
| Spread RMSE | 1.6173 | 1.6418 |

## Observations

- Classical Elo achieves a lower spread MAE by 0.0169, suggesting better calibrated point spread predictions.
- Classical Elo shows higher win/loss prediction accuracy on decisive games.
- The Bayesian model provides full three-way probability estimates (H/D/A), enabling categorical predictions and proper log-loss evaluation.
- Both models update over time: Classical Elo updates per-match (online), while Bayesian Elo retrains per-season on expanding windows of data. This is a fairer comparison than a single held-out season.
