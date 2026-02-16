# CLAUDE.md

## Incremental Usage Pattern

The Bayesian Elo system can be used in an incremental predict-then-update loop
using the existing building blocks in `bayesian_elo.py`.

### Step 1: Initial fit on historical data

```python
historical_df = ...  # columns: season, date, home_team, away_team, result
data = prepare_data(historical_df)
model = build_model(data)
trace = fit_model(model)
```

### Step 2: Predict upcoming games

`predict_outcomes` computes probabilities entirely from the posterior samples
(`theta`, `alpha*`, `beta*`) and team/period indices. It only references
`data["outcome"]` at the end to fill the `actual_outcome` column. This means
you can construct a synthetic data dict for games with unknown outcomes:

```python
upcoming_games = [...]  # list of (home_team, away_team) tuples
future_data = {
    "home_idx": np.array([data["team_to_idx"][h] for h, a in upcoming_games]),
    "away_idx": np.array([data["team_to_idx"][a] for h, a in upcoming_games]),
    "period_idx": np.array([data["n_periods"] - 1] * len(upcoming_games)),
    "outcome": np.zeros(len(upcoming_games), dtype=int),  # dummy
    "n_teams": data["n_teams"],
    "n_periods": data["n_periods"],
}
predictions = predict_outcomes(trace, future_data)
# Use p_home_win, p_draw, p_away_win; ignore actual_outcome
```

`period_idx` is set to the last period, meaning "use the most recent strength
estimates." This is appropriate when upcoming games are in the current or next
season.

### Step 3: Incorporate results and refit

```python
new_results_df = ...  # completed games with actual results
updated_df = pd.concat([historical_df, new_results_df], ignore_index=True)
data = prepare_data(updated_df)
model = build_model(data)
trace = fit_model(model)
# Go back to Step 2
```

### Limitations and alternatives

The model refits from scratch each cycle because `build_model` bakes observed
data into the PyMC graph (line 163: `observed=outcome`). Three options if
refitting becomes too slow:

1. **Just refit each time.** For modest datasets (a few thousand games, ~20
   teams), MCMC completes in minutes. Simplest path.

2. **Use the previous posterior as an informative prior.** Extract posterior
   summaries from the previous trace (means/stds of `theta` for the last
   period) and pass them as tighter priors to a modified `build_model` that
   only covers the new period. This would require `build_model` to accept
   optional prior parameters for `theta_init`.

3. **Switch to variational inference.** Replace `pm.sample()` with `pm.fit()`
   (ADVI) in `fit_model`. Faster but less accurate in the posterior tails.

### Note on fit_model

`fit_model` draws samples from the posterior via MCMC (NUTS), not optimization.
The returned `trace` is an `arviz.InferenceData` containing thousands of
posterior samples for every parameter. Predictions average over these samples
to produce posterior mean probabilities.
