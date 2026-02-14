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

