# Glickman (2025) Comparison of Bayesian vs Elo: Analysis and EPL Application

## How Glickman Compared the Two Approaches

### The Problem with Traditional Elo

Traditional Elo treats a draw as "half a win plus half a loss" -- it has no
mechanism to model draw probability as its own outcome. This works passably when
draws are uncommon, but breaks down in settings with high draw rates. In ICCF
correspondence chess, ~95% of games between top players end in draws, causing
**rating stagnation** where elite players' ratings barely move because the system
cannot distinguish between an expected draw and an informative one.

### The Bayesian Model's Key Innovation

The Glickman (2025) model (Eq 3.1) gives each outcome (win/draw/loss) its own
probability via a multinomial logit, where draw probability depends on the
*average strength* of the two players through parameters:

- `beta_0`: baseline draw rate
- `beta_1`: strength-dependent draw rate (higher-strength pairs draw more)

This captures the empirical reality that stronger players draw more often.

### Evaluation Method: Cross-Entropy on Held-Out Predictions (Section 6)

Glickman split the data into training (20 periods) and validation (5 periods,
17,414 games). For each validation game, one-step-ahead predictive probabilities
were computed using Gauss-Hermite quadrature over both players' rating
distributions.

The primary metric was **cross-entropy** (negative predictive log-likelihood per
game):

| System | Cross-Entropy | Reduction vs Baseline |
|--------|--------------|----------------------|
| Random baseline (empirical frequencies) | 0.8126 | -- |
| Bayesian (ICCF-implemented params) | 0.5724 | 29.6% |
| Bayesian (optimized params) | 0.5063 | 37.7% |

The baseline assigned probabilities proportional to empirical outcome
frequencies: 29.6% decisive (split equally between win/loss) and 70.4% draws.

### Calibration Checks (Figure 4)

Glickman also showed:
1. Predicted draw probabilities are higher for games that actually drew vs
   decisive games (boxplot comparison).
2. For decisive games, the model assigned the correct winner >50% probability
   85.2% of the time.

### The "Optimized vs Practical" Tension

The statistically optimal parameters produced ratings that were too volatile and
inflated for practical use (top players exceeding 3000 Elo). The ICCF chose
suboptimal-but-reasonable parameters instead:

- Optimized `tau = 0.460` (~80 Elo points volatility) vs implemented
  `tau = 0.144` (~25 Elo points)
- Optimized draw rates implied 41.6% draws at 1500 Elo, 95% at 2500 Elo
- Implemented draw rates: 60% at 1500 Elo, 80% at 2500 Elo

The ICCF-implemented system traded 8 percentage points of cross-entropy
reduction for interpretable, stable ratings.

## Applying This to EPL Football

### Cross-Entropy Evaluation Design

We replicate Glickman's evaluation using our expanding-window infrastructure.
For each test season s (from 2nd to last):

1. **Bayesian model:** Train on seasons 1..s-1, predict season s. The model
   directly outputs 3-way probabilities P(H), P(D), P(A).

2. **Classical Elo:** Run sequentially through all prior matches. Elo produces
   only a home-win probability P(H|Elo). To compute 3-way cross-entropy, we
   convert to 3-way probabilities using the empirical draw rate from training:
   - P(D) = empirical draw rate from seasons 1..s-1
   - P(H) = (1 - P(D)) * P(H|Elo)
   - P(A) = (1 - P(D)) * (1 - P(H|Elo))

3. **Naive baseline:** Assign empirical outcome frequencies from training data
   (seasons 1..s-1) as the predicted probability for every game.

The cross-entropy for each model is:
  CE = -mean(log P(actual outcome))

### Key Differences from Glickman's Setting

- **Home advantage matters.** The ICCF voted to exclude white advantage
  (alpha_0 = alpha_1 = 0). In EPL, home advantage is significant and our model
  includes it via alpha_0 and alpha_1.

- **Draw rates are lower and less strength-dependent.** EPL draw rates are
  ~25-27%, compared to 70%+ in ICCF chess. The strength-dependence of draws
  is less extreme in football than in engine-assisted chess, though it is worth
  checking empirically.

- **Season structure.** EPL has a clear season structure with promotion/
  relegation, introducing team turnover. The Bayesian model handles unseen
  teams by assigning mean-strength priors (theta=0).

### Expected Insights

The cross-entropy comparison will show whether the Bayesian model's explicit
3-way probability modeling provides better-calibrated predictions than Elo's
implicit draw handling. Even if the absolute improvement is modest (EPL draws
are less extreme than ICCF), the Bayesian model should provide:

1. Better cross-entropy through explicit draw probability modeling
2. Strength-dependent draw rates (if beta_1 > 0 in the posterior)
3. Uncertainty quantification (posterior intervals on ratings and predictions)
