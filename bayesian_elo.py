"""
Bayesian Elo rating system using PyMC.

Implements the Glickman (2025) model for rating competitors in games with
strength-dependent draw probabilities. Uses PyMC/NUTS for full posterior
inference rather than the approximate filtering algorithm from the paper.

Model (Eq 3.1):
    P(home win)  ~ exp(theta_home + x*(alpha0 + alpha1*avg_theta)/4)
    P(away win)  ~ exp(theta_away - x*(alpha0 + alpha1*avg_theta)/4)
    P(draw)      ~ exp(beta0 + (1 + beta1)*avg_theta)
    Normalized via softmax.

Time evolution (Eq 3.2):
    theta_{i,t+1} ~ N(theta_{i,t}, tau^2)

Rating conversion (Eq 6.1):
    R = 1500 + 173.72 * theta
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az  # noqa: F401 — used by callers via this module


RATING_SCALE = 173.72  # 400 / log(10), per Eq 6.1
RATING_BASELINE = 1500


def prepare_data(df):
    """Prepare match data for the Bayesian model.

    Maps teams to integer indices, seasons to period indices, and encodes
    outcomes as integers for pm.Categorical (H=2, D=1, A=0).

    Parameters
    ----------
    df : DataFrame
        Must contain columns: season, home_team, away_team, result (H/D/A).

    Returns
    -------
    dict with keys: home_idx, away_idx, period_idx, outcome, n_teams,
    n_periods, team_names, season_names, team_to_idx, season_to_idx
    """
    df = df.sort_values(["season", "date"]).reset_index(drop=True)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_to_idx = {t: i for i, t in enumerate(teams)}

    seasons = sorted(df["season"].unique())
    season_to_idx = {s: i for i, s in enumerate(seasons)}

    outcome_map = {"A": 0, "D": 1, "H": 2}

    home_idx = np.array([team_to_idx[t] for t in df["home_team"]])
    away_idx = np.array([team_to_idx[t] for t in df["away_team"]])
    period_idx = np.array([season_to_idx[s] for s in df["season"]])
    outcome = np.array([outcome_map[r] for r in df["result"]])

    return {
        "home_idx": home_idx,
        "away_idx": away_idx,
        "period_idx": period_idx,
        "outcome": outcome,
        "n_teams": len(teams),
        "n_periods": len(seasons),
        "team_names": teams,
        "season_names": seasons,
        "team_to_idx": team_to_idx,
        "season_to_idx": season_to_idx,
    }


def build_model(data):
    """Build the PyMC model implementing the Glickman (2025) framework.

    The model uses GaussianRandomWalk for latent team strengths evolving
    over seasons, and a Categorical likelihood with multinomial logit
    probabilities for win/draw/loss outcomes.

    Parameters
    ----------
    data : dict
        Output from prepare_data().

    Returns
    -------
    pm.Model
    """
    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    period_idx = data["period_idx"]
    outcome = data["outcome"]
    n_teams = data["n_teams"]
    n_periods = data["n_periods"]

    with pm.Model() as model:
        # --- System parameters (weakly informative priors) ---
        # Draw parameters
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        beta1 = pm.HalfNormal("beta1", sigma=0.5)

        # Home advantage parameters (x=1 for home team in football)
        alpha0 = pm.Normal("alpha0", mu=0.3, sigma=0.5)
        alpha1 = pm.Normal("alpha1", mu=0, sigma=0.3)

        # Random walk volatility
        tau = pm.HalfNormal("tau", sigma=0.5)

        # Initial strength spread
        sigma_init = pm.HalfNormal("sigma_init", sigma=1.0)

        # --- Latent team strengths ---
        # Non-centered parameterization for better sampling:
        # theta_raw ~ Normal(0, 1), theta = cumsum(theta_raw * scale)
        theta_init_raw = pm.Normal(
            "theta_init_raw", mu=0, sigma=1, shape=n_teams
        )
        theta_init = pm.Deterministic(
            "theta_init", theta_init_raw * sigma_init
        )

        if n_periods > 1:
            theta_innovations_raw = pm.Normal(
                "theta_innovations_raw",
                mu=0,
                sigma=1,
                shape=(n_periods - 1, n_teams),
            )
            theta_innovations = theta_innovations_raw * tau

            # Build full theta array: shape (n_periods, n_teams)
            theta_list = [theta_init]
            for t in range(n_periods - 1):
                theta_list.append(theta_list[-1] + theta_innovations[t])
            theta = pm.Deterministic("theta", pt.stack(theta_list, axis=0))
        else:
            theta = pm.Deterministic("theta", theta_init[None, :])

        # --- Extract strengths for each match ---
        theta_home = theta[period_idx, home_idx]
        theta_away = theta[period_idx, away_idx]
        avg_theta = (theta_home + theta_away) / 2

        # --- Multinomial logit probabilities (Eq 3.1) ---
        # x = 1 for home team (football convention: home = white)
        x = 1.0

        # Log-unnormalized probabilities
        log_p_home_win = theta_home + x * (alpha0 + alpha1 * avg_theta) / 4
        log_p_away_win = theta_away - x * (alpha0 + alpha1 * avg_theta) / 4
        log_p_draw = beta0 + (1 + beta1) * avg_theta

        # Stack as (n_matches, 3): columns = [away_win, draw, home_win]
        logits = pt.stack(
            [log_p_away_win, log_p_draw, log_p_home_win], axis=1
        )

        # --- Likelihood ---
        pm.Categorical("obs", logit_p=logits, observed=outcome)

    return model


def fit_model(model, draws=2000, tune=2000, chains=4, target_accept=0.9,
              random_seed=42):
    """Run MCMC sampling on the model.

    Parameters
    ----------
    model : pm.Model
    draws : int
        Number of posterior draws per chain.
    tune : int
        Number of tuning steps per chain.
    chains : int
        Number of MCMC chains.
    target_accept : float
        Target acceptance rate for NUTS.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    arviz.InferenceData
    """
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
    return trace


def extract_ratings(trace, data, period=-1):
    """Convert posterior theta samples to Elo-scale ratings.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from fit_model().
    data : dict
        Output from prepare_data().
    period : int
        Which time period to extract ratings for (-1 = last).

    Returns
    -------
    DataFrame with columns: team, rating_mean, rating_std, ci_lower, ci_upper
    """
    # theta has shape (chains, draws, n_periods, n_teams)
    theta_samples = trace.posterior["theta"].values
    # Reshape to (n_samples, n_periods, n_teams)
    n_chains, n_draws = theta_samples.shape[:2]
    theta_flat = theta_samples.reshape(n_chains * n_draws,
                                       data["n_periods"],
                                       data["n_teams"])

    # Select the requested period
    theta_period = theta_flat[:, period, :]  # (n_samples, n_teams)

    # Convert to Elo scale
    ratings = RATING_BASELINE + RATING_SCALE * theta_period

    rating_mean = ratings.mean(axis=0)
    rating_std = ratings.std(axis=0)
    ci_lower = np.percentile(ratings, 2.5, axis=0)
    ci_upper = np.percentile(ratings, 97.5, axis=0)

    df = pd.DataFrame({
        "team": data["team_names"],
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }).sort_values("rating_mean", ascending=False).reset_index(drop=True)

    return df


def predict_outcomes(trace, data):
    """Compute posterior mean outcome probabilities for each match.

    Parameters
    ----------
    trace : arviz.InferenceData
    data : dict

    Returns
    -------
    DataFrame with columns: p_away_win, p_draw, p_home_win, pred_outcome,
    actual_outcome
    """
    theta_samples = trace.posterior["theta"].values
    n_chains, n_draws = theta_samples.shape[:2]
    theta_flat = theta_samples.reshape(n_chains * n_draws,
                                       data["n_periods"],
                                       data["n_teams"])

    # System parameter samples
    beta0 = trace.posterior["beta0"].values.flatten()
    beta1 = trace.posterior["beta1"].values.flatten()
    alpha0 = trace.posterior["alpha0"].values.flatten()
    alpha1 = trace.posterior["alpha1"].values.flatten()

    home_idx = data["home_idx"]
    away_idx = data["away_idx"]
    period_idx = data["period_idx"]
    n_matches = len(home_idx)
    n_samples = len(beta0)

    # Compute probabilities for each sample, then average
    # Process in batches to manage memory
    batch_size = 200
    p_accum = np.zeros((n_matches, 3))

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_theta = theta_flat[start:end]  # (batch, n_periods, n_teams)
        batch_beta0 = beta0[start:end]
        batch_beta1 = beta1[start:end]
        batch_alpha0 = alpha0[start:end]
        batch_alpha1 = alpha1[start:end]

        # (batch, n_matches)
        th_home = batch_theta[:, period_idx, home_idx]
        th_away = batch_theta[:, period_idx, away_idx]
        avg_th = (th_home + th_away) / 2

        x = 1.0
        b0 = batch_beta0[:, None]
        b1 = batch_beta1[:, None]
        a0 = batch_alpha0[:, None]
        a1 = batch_alpha1[:, None]

        log_p_hw = th_home + x * (a0 + a1 * avg_th) / 4
        log_p_aw = th_away - x * (a0 + a1 * avg_th) / 4
        log_p_d = b0 + (1 + b1) * avg_th

        # Softmax
        logits = np.stack([log_p_aw, log_p_d, log_p_hw], axis=2)
        logits_max = logits.max(axis=2, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)

        p_accum += probs.sum(axis=0)

    p_mean = p_accum / n_samples

    outcome_labels = {0: "A", 1: "D", 2: "H"}
    pred_outcome = [outcome_labels[i] for i in np.argmax(p_mean, axis=1)]
    actual_outcome = [outcome_labels[o] for o in data["outcome"]]

    return pd.DataFrame({
        "p_away_win": p_mean[:, 0],
        "p_draw": p_mean[:, 1],
        "p_home_win": p_mean[:, 2],
        "pred_outcome": pred_outcome,
        "actual_outcome": actual_outcome,
    })


def evaluate_predictions(predictions):
    """Evaluate prediction quality.

    Parameters
    ----------
    predictions : DataFrame
        Output from predict_outcomes().

    Returns
    -------
    dict with categorical_accuracy, log_loss, decisive_accuracy
    """
    n = len(predictions)

    # Categorical accuracy
    correct = (predictions["pred_outcome"] == predictions["actual_outcome"])
    categorical_accuracy = correct.mean()

    # Log-loss (cross-entropy)
    eps = 1e-10
    log_loss_vals = []
    for _, row in predictions.iterrows():
        if row["actual_outcome"] == "A":
            p = row["p_away_win"]
        elif row["actual_outcome"] == "D":
            p = row["p_draw"]
        else:
            p = row["p_home_win"]
        log_loss_vals.append(-np.log(np.clip(p, eps, 1.0)))
    log_loss = np.mean(log_loss_vals)

    # Accuracy on decisive (non-draw) games
    decisive = predictions[predictions["actual_outcome"] != "D"]
    if len(decisive) > 0:
        decisive_correct = (
            decisive["pred_outcome"] == decisive["actual_outcome"]
        )
        decisive_accuracy = decisive_correct.mean()
    else:
        decisive_accuracy = float("nan")

    return {
        "categorical_accuracy": categorical_accuracy,
        "log_loss": log_loss,
        "decisive_accuracy": decisive_accuracy,
        "n_matches": n,
    }
