# Draw Prevalence Analysis

Analysis of draw frequency in 11,113 Premier League matches (1993–2022) and
whether higher-rated matchups produce more draws.

## How to run

```bash
uv run python draw_analysis.py
```

Outputs three charts to `data/` and prints all statistics to stdout.

## Method

Each match is assigned pre-match Elo ratings for both teams using the tuned
classical Elo system (K=3, home advantage=20, spread factor=55). Two measures
are derived per match:

- **Average Elo** — mean of home and away ratings; higher values indicate both
  teams are strong ("match quality").
- **Elo difference** — absolute rating gap; smaller values indicate evenly
  matched teams ("competitiveness").

Draw rates are then compared across quintile bins, a 2D cross-tabulation, and
a logistic regression. The first season (1993-94) is excluded from
rating-based analyses because all teams start at 1500.

## Results

### Overall draw prevalence

| Outcome   | Count | Rate  |
|-----------|-------|-------|
| Home win  | 5,088 | 45.8% |
| Draw      | 2,864 | 25.8% |
| Away win  | 3,161 | 28.4% |

Season-level draw rates range from 18.7% (2018-19) to 31.3% (1996-97).

### Draw rate by average Elo (match quality)

| Quintile | Elo range   | Draw rate |
|----------|-------------|-----------|
| Q1 (lowest)  | 1466–1494 | 28.1% |
| Q2           | 1494–1501 | 28.5% |
| Q3           | 1501–1512 | 26.4% |
| Q4           | 1512–1525 | 22.8% |
| Q5 (highest) | 1525–1587 | 22.0% |

Point-biserial correlation: r = −0.055, p < 0.0001.

Higher-rated matchups draw *less* often, not more.

### Draw rate by Elo difference (competitiveness)

| Quintile | Diff range | Draw rate |
|----------|------------|-----------|
| Q1 (closest)       | 0–6   | 29.3% |
| Q2                  | 6–14  | 28.9% |
| Q3                  | 14–27 | 27.1% |
| Q4                  | 27–50 | 24.4% |
| Q5 (biggest gap)    | 50–125| 18.0% |

Point-biserial correlation: r = −0.098, p < 0.0001.

Evenly matched teams draw substantially more often.

### 2D cross-tabulation (quality × competitiveness)

|           | Close | Mid   | Mismatch |
|-----------|-------|-------|----------|
| Low Elo   | 29.8% | 27.2% | 26.6%   |
| Mid Elo   | 28.1% | 27.1% | 22.1%   |
| High Elo  | 28.3% | 27.6% | 19.4%   |

Within every quality tier, closer matches draw more. The competitiveness
effect dominates the quality effect.

### Traditional Top 6

| Match type         | Draw rate | Matches |
|--------------------|-----------|---------|
| Top 6 vs Top 6     | 26.9%    | 784     |
| Top 6 vs Non-Top 6 | 22.1%    | 4,606   |
| Other              | 28.4%    | 5,261   |

Chi-squared test: χ² = 51.68, p < 0.0001. Top-6 teams playing weaker
opponents draw the least — they tend to win decisively.

### Logistic regression

P(draw) ~ avg\_elo + elo\_diff (standardized features):

| Predictor | Coefficient |
|-----------|-------------|
| avg_elo   | +0.003      |
| elo_diff  | −0.241      |
| intercept | −1.083      |

With both predictors included, average Elo has near-zero effect. Match
competitiveness (closeness of ratings) is what drives draw probability.

## Conclusion

Draws in the Premier League are primarily driven by how evenly matched the two
teams are, not by how strong they are in absolute terms. Matches between two
highly-rated teams actually draw slightly less often than average, because
high-Elo matchups are disproportionately mismatches (a top team hosting a
mid-table side). When controlling for competitiveness, absolute quality has
essentially no effect on draw probability.
