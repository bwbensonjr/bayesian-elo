# Training Window Depth Experiment

## Question

Does including more historical seasons improve Bayesian model predictions, or
does older data add noise? We hold out 2021-22 as the test season and compare
cross-entropy across three training window depths.

## Setup

| Depth | Training Seasons | Matches |
|-------|-----------------|---------|
| All prior seasons | 1993-94 through 2020-21 (28 seasons) | 10,804 |
| Half | 2007-08 through 2020-21 (14 seasons) | 5,320 |
| One season | 2020-21 only (1 season) | 380 |

Each configuration runs:

1. **Bayesian model** with weekly in-season refits on 2021-22 (initial fit:
   4 chains × 1000 draws, weekly refits: 2 chains × 500 draws)
2. **Classical Elo** on the same training slice plus test season (sequential
   online updates, tuned parameters K=3, home_field=20, spread_factor=55)
3. **Naive baseline** using empirical H/D/A frequencies from the training slice

Elo runs on `concat(train_slice, test_season)` so it sees exactly the same
historical information as the Bayesian model for each depth.

## Results

Cross-entropy (negative log-likelihood per game, lower is better):

| Depth | Train Matches | Baseline | Elo | Bayesian | Bayes−Elo |
|-------|--------------|----------|------|----------|-----------|
| All (28 seasons) | 10,804 | 1.0805 | 1.0531 | 0.9689 | −0.0843 |
| Half (14 seasons) | 5,320 | 1.0764 | 1.0518 | **0.9667** | −0.0851 |
| 1 season | 380 | 1.0777 | 1.0611 | 0.9931 | −0.0680 |

## Interpretation

### Half the data is slightly better than all of it

The 14-season window (2007-08 onward) achieves the lowest Bayesian
cross-entropy at 0.9667, marginally better than the full 28-season window
(0.9689). The difference is small (0.0022), but the direction is consistent
with the idea that pre-2007 Premier League data reflects a different
competitive landscape — different teams, different playing styles, and
different home-advantage patterns — that adds slight noise to the hierarchical
model's estimates.

### One season is too little

With only 380 training matches from 2020-21, the Bayesian model's
cross-entropy rises to 0.9931. The hierarchical structure needs more than one
season of data to learn stable team-strength dynamics and the draw/home-
advantage parameters (beta_0, beta_1, alpha_0, alpha_1). Interestingly, the
Bayesian model still beats Elo even with a single training season, but the
margin shrinks from ~0.085 to ~0.068.

### Elo is relatively stable across depths

Elo cross-entropy varies little across depths (1.0518–1.0611), because its
sequential online updates naturally downweight old information through
continuous rating adjustments. The main effect of a shorter training window
for Elo is less accurate initial ratings at the start of 2021-22.

### The Bayesian advantage is robust

Across all three depths, the Bayesian model outperforms Elo by 0.068–0.085
cross-entropy units. The advantage is largest when the Bayesian model has
enough data to estimate its hierarchical parameters well (14+ seasons) and
smallest with a single training season.

## Practical Implications

For the Premier League use case:

- **Use 10–15 seasons of training data** for the best balance between signal
  and noise. Going back to the early 1990s provides diminishing (or slightly
  negative) returns.
- **One season is not enough** for initial model fitting, though weekly refits
  during the season partially compensate.
- **The choice is not critical.** The difference between "all" and "half" is
  small enough that either is a reasonable default. The bigger gain comes from
  using the Bayesian model at all rather than Elo.

## Reproducing

```bash
# Full run (default MCMC settings, ~90 minutes)
python premier_league.py --depth-experiment --weekly

# Quick smoke test (~75 minutes with minimal settings)
python premier_league.py --depth-experiment --weekly \
  --initial-draws 100 --initial-tune 100 --initial-chains 2 \
  --weekly-draws 100 --weekly-tune 100 --weekly-chains 2
```

Results are saved to `data/depth-experiment.csv` (per-game) and
`data/depth-experiment-summary.csv` (summary table).
