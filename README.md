# Bayesian Elo

Mark Glickman has invented a Bayesian approach to Chess Elo ratings
that is better at dealing with the preponderance of draws in
high-level chess.

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

### Simplifications vs. the Paper

- **No Gauss-Hermite quadrature or Newton-Raphson** — PyMC/NUTS
  samples the full joint posterior directly
- **No opponent prior approximation** — all team strengths are
  estimated simultaneously
- **System parameters estimated jointly** with latent strengths
  (fully Bayesian) rather than optimized separately via predictive
  likelihood

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

