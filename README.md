# Bayesian Elo

Mark Glickman has invented a Bayesian approach to Chess Elo ratings
that is better at dealing with the preponderance of draws in
high-level chess.

Harvard Gazette article: [Breaking chess's ratings
stalemate](https://news.harvard.edu/gazette/story/2026/02/breaking-chesss-rating-stalemate/)

[Mark E. Glickman, Rating Competitors in Games with Strength-Dependent
Tie Probabilities, J. data sci.(2025), 1-20, DOI
10.6339/25-JDS1209](https://jds-online.org/journal/JDS/article/1455/info)

* [PDF](paper/jds1209.pdf)
* [Markdown](paper/jds1209.md)

The goal of this exercise is to test out the approach as applied to
Premier League football which also has a reasonable number of
draws/tie games.

We will first implement a simple Elo-based rating in Python and
measure its accuracy. We will then implement the Glickman Bayesian
method and compare its accuracy to the traditional Elo rating.

## Test Data 

We will use the 11,113 Premier League matches from the 1992-93 season
through the 2021-22 season in the
`data/premier-league-1993-2022.parquet` file and read using
`premier_league.read_premier_results`.

## Elo Methodology 

The Elo harness will be based on the implementation in

https://github.com/bwbensonjr/high-school-sports/blob/main/elo.py

with the parameters tuned for the Premier League results in the test
data set.

### Tuned Parameters

A two-stage grid search (coarse then fine) over the Elo parameters
minimized the RMSE of the predicted goal spread versus the actual goal
spread across all 11,113 matches:

| Parameter | Value |
|-----------|-------|
| `k` (K-factor) | 3 |
| `home_field` (home advantage in Elo points) | 20 |
| `spread_factor` (Elo-to-goals divisor) | 55 |

### Results

| Metric | Value |
|--------|-------|
| RMSE (predicted vs. actual goal spread) | 1.617 |
| MAE (predicted vs. actual goal spread) | 1.247 |
| Win prediction accuracy (excl. draws) | 69.9% |

The final Elo rankings place Man City, Liverpool, Chelsea, Man United,
and Tottenham as the top five — consistent with historical Premier
League performance over this period.

## Bayesian Elo Methodology

The Bayesian approach implements the Glickman (2025) state-space model
using [PyMC](https://www.pymc.io/) for full posterior inference via
NUTS (No U-Turn Sampler), rather than the approximate filtering
algorithm (Gauss-Hermite + Newton-Raphson) described in the paper.

### Model

**Outcome model (Eq 3.1):** Multinomial logit over win/draw/loss:

- log P(home win) = θ\_home + (α₀ + α₁ · avg\_θ) / 4
- log P(away win) = θ\_away − (α₀ + α₁ · avg\_θ) / 4
- log P(draw) = β₀ + (1 + β₁) · avg\_θ

Probabilities are normalized via softmax. The key feature is that
**draw probability increases with average team strength** (controlled
by β₁), matching the empirical observation that stronger teams draw
more often.

**Time evolution (Eq 3.2):** Team strengths follow a Gaussian random
walk across seasons: θ\_{i,t+1} ~ N(θ\_{i,t}, τ²), implemented via a
non-centered parameterization for efficient NUTS sampling.

**Rating conversion (Eq 6.1):** R = 1500 + 173.72 · θ

### Differences from the Paper's Filtering Algorithm

- **No Gauss-Hermite quadrature or Newton-Raphson** — PyMC/NUTS
  samples the full joint posterior directly
- **No opponent prior approximation** — all team strengths are
  estimated simultaneously
- **System parameters estimated jointly** with latent strengths
  (fully Bayesian) rather than optimized separately via predictive
  likelihood

See the [Glickman Filter](#glickman-filter) section below for an
implementation of the paper's approximate filtering algorithm.

### Results

MCMC sampling: 4 chains × 4,000 iterations (2,000 tune + 2,000
draws), 0 divergences, ~1,450 latent parameters (50 teams × 29
seasons).

| Metric | Value |
|--------|-------|
| Categorical accuracy (H/D/A) | 53.7% |
| Log-loss | 0.965 |
| Win prediction accuracy (excl. draws) | 72.3% |
| Actual draw rate | 25.8% |
| Mean predicted P(draw) | 25.8% |

The Bayesian model provides **uncertainty intervals** for each team's
rating. Top 10 rankings for the final season (2021-22), with 95%
credible intervals:

| Rank | Team | Rating | 95% CI |
|------|------|--------|--------|
| 1 | Man City | 1956 | [1820, 2097] |
| 2 | Liverpool | 1922 | [1785, 2063] |
| 3 | Chelsea | 1792 | [1657, 1927] |
| 4 | Man United | 1743 | [1611, 1879] |
| 5 | Tottenham | 1720 | [1591, 1850] |
| 6 | Arsenal | 1693 | [1564, 1825] |
| 7 | Leicester | 1638 | [1506, 1767] |
| 8 | West Ham | 1623 | [1496, 1752] |
| 9 | Wolves | 1591 | [1461, 1720] |
| 10 | Leeds | 1540 | [1400, 1678] |

## Comparison

| Metric | Classical Elo | Bayesian |
|--------|---------------|----------|
| Win accuracy (excl. draws) | 69.9% | **72.3%** |
| Categorical accuracy (H/D/A) | N/A | 53.7% |
| Log-loss | N/A | 0.965 |
| Point spread MAE | 1.247 | N/A |

The Bayesian model improves decisive game prediction by 2.4 percentage
points, and its draw calibration is essentially perfect (25.8%
predicted vs. 25.8% actual).

## Glickman Filter

`glickman_filter.py` implements the approximate filtering algorithm
from Sections 4.1-4.3 of the paper. This provides closed-form
posterior updates with no PyMC dependency, making it orders of
magnitude faster than MCMC for the same model.

### Algorithm

The filter processes games period-by-period (one season per period):

1. **Opponent-prior approximation (Section 4.1):** Within each period,
   all players are updated using opponents' *prior* distributions
   (from the start of the period), making updates independent across
   players.

2. **2-point Gauss-Hermite quadrature (Section 4.2):** The integral
   over opponent strength is approximated by evaluating the likelihood
   at μ\_j ± σ\_j with equal weights 1/2.

3. **One-step Newton-Raphson (Section 4.3):** The posterior for each
   player is approximated as Gaussian with mean μ\* = μ − g/h and
   variance σ\*² = −1/h, where g and h are the gradient and Hessian
   of the log-posterior evaluated at the prior mean.

4. **Period advance:** σ\_prior = min(√(σ\_post² + τ²), σ\_cap)

### System Parameters

System parameters (`alpha0`, `alpha1`, `beta0`, `beta1`, `tau`,
`sigma_init`, `sigma_cap`) are treated as fixed inputs, not sampled.
They can be optimized via one-step-ahead predictive log-likelihood
(Section 5) using `optimize_params()`.

### Results

#### Default parameters

With hand-picked default parameters, the filter underperforms MCMC on
all metrics (in-sample over all 11,113 matches):

| Metric | MCMC | Filter (default) |
|--------|------|------------------|
| Categorical accuracy (H/D/A) | 53.7% | 49.4% |
| Log-loss | 0.965 | 1.019 |
| Win prediction accuracy (excl. draws) | 72.3% | 66.6% |

The gap is primarily because MCMC jointly estimates system parameters
from the data while the filter uses fixed defaults.

#### Optimized parameters

Running `optimize_params()` on the full dataset maximizes the
one-step-ahead predictive log-likelihood (Section 5) via Nelder-Mead:

| Parameter | Default | Optimized | Description |
|-----------|---------|-----------|-------------|
| `alpha0` | 0.21 | **1.14** | Home advantage intercept |
| `alpha1` | 0.0 | 0.009 | Strength-dependent home advantage |
| `beta0` | -0.39 | **-0.20** | Draw intercept |
| `beta1` | 0.12 | ~0 | Strength-dependent draw tendency |
| `tau` | 0.15 | **0.24** | Season-to-season volatility |
| `sigma_init` | 0.50 | **0.35** | Initial rating uncertainty |

The largest shifts are in `alpha0` (home advantage much stronger than
the default) and `tau` (more season-to-season volatility). The
strength-dependent draw parameter `beta1` optimized to near zero,
suggesting this effect is not significant in Premier League data.

#### Expanding-window out-of-sample comparison

To compare fairly across all three approaches, each model predicts
10,651 matches across seasons 1994-95 through 2021-22 using only
data from prior seasons (expanding-window evaluation):

| Metric | Elo | MCMC | Filter (default) | Filter (optimized) |
|--------|-----|------|-------------------|--------------------|
| Categorical accuracy (H/D/A) | -- | 51.6% | 49.8% | **51.5%** |
| Win accuracy (excl. draws) | **70.2%** | 69.3% | 66.8% | 69.1% |
| Log-loss | 1.047 | 0.999 | 1.015 | **0.997** |

With optimized parameters, the filter matches MCMC across all
metrics. On log-loss it edges out MCMC (0.997 vs 0.999), achieving
the lowest log-loss in 20 of 28 individual seasons. Both Bayesian
approaches substantially outperform classical Elo on log-loss, which
measures the full three-way probability calibration.

Classical Elo retains the highest decisive-game win accuracy (70.2%),
likely because its per-match online updates adapt faster within a
season than the filter's per-season batch updates.

### Filter Usage

```python
from bayesian_elo import prepare_data
from glickman_filter import SystemParams, run_filter, predict_outcomes, extract_ratings

df = ...  # columns: season, date, home_team, away_team, result
data = prepare_data(df)
params = SystemParams()

state = run_filter(data, params)
ratings = extract_ratings(state, data, period=-1)
predictions = predict_outcomes(state, data, params)
```

The `GlickmanPredictor` class provides a stateful wrapper for
incremental predict-then-update cycles:

```python
from glickman_filter import GlickmanPredictor

predictor = GlickmanPredictor(historical_df)
predictor.fit()
preds = predictor.predict(upcoming_games_df)
predictor.update(completed_games_df)
```

## Usage

Run both systems with pre-tuned parameters (default, no grid search):

```
uv run python premier_league.py
```

Run only the classical Elo system:

```
uv run python premier_league.py --elo-only
```

Run only the Bayesian system:

```
uv run python premier_league.py --bayesian-only
```

Re-run the Elo parameter grid search:

```
uv run python premier_league.py --tune
```

