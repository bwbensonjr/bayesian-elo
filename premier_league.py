import argparse

import numpy as np
import pandas as pd
from itertools import product

from elo import Elo
from bayesian_elo import (
    prepare_data,
    build_model,
    fit_model,
    extract_ratings,
    predict_outcomes,
    evaluate_predictions,
)

# Best Elo parameters from grid search on 1993-2022 Premier League data
TUNED_ELO_PARAMS = {"k": 3, "home_field": 20, "spread_factor": 55}


def read_premier_results():
    """Read Premier League results and transform into desired format."""
    df = (
        pd.read_parquet("data/premier-league-1993-2022.parquet")
        [[
            "Season",
            "DateTime",
            "HomeTeam",
            "AwayTeam",
            "FTHG",
            "FTAG",
            "FTR"
        ]]
        .rename(columns={
            "Season": "season",
            "DateTime": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_team_goals",
            "FTAG": "away_team_goals",
            "FTR": "result",
        })
        .assign(
            date=lambda x: x["date"].dt.date,
            actual_spread=lambda x: x["home_team_goals"] - x["away_team_goals"],
        )
    )
    return df


def compute_spread_calibration(df):
    """Compute conditional average spreads from historical data.

    Returns dict with mean actual spread for each outcome category:
    - "H": mean spread when home team won
    - "D": mean spread when draw
    - "A": mean spread when away team won
    """
    return {
        "H": df.loc[df["result"] == "H", "actual_spread"].mean(),
        "D": df.loc[df["result"] == "D", "actual_spread"].mean(),
        "A": df.loc[df["result"] == "A", "actual_spread"].mean(),
    }


def group_games_by_week(season_df):
    """Group a season's games by ISO calendar week.

    Parameters
    ----------
    season_df : DataFrame
        Games for a single season with a 'date' column (date objects).

    Returns
    -------
    list of (week_label, DataFrame) tuples sorted chronologically.
    Week labels use ISO format '%G-W%V'.
    """
    season_df = season_df.sort_values("date").copy()
    season_df["_week"] = season_df["date"].apply(
        lambda d: f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
    )
    weeks = []
    for week_label, group in season_df.groupby("_week", sort=False):
        weeks.append((week_label, group.drop(columns=["_week"])))
    # Sort by the earliest date in each group
    weeks.sort(key=lambda x: x[1]["date"].iloc[0])
    return weeks


def bayesian_spread_predictions(predictions, calibration, df):
    """Derive expected point spreads from Bayesian win/draw/loss probabilities.

    Uses E[spread] = P(H)*E[spread|H] + P(D)*E[spread|D] + P(A)*E[spread|A].
    """
    predictions = predictions.copy()
    predictions["pred_spread"] = (
        predictions["p_home_win"] * calibration["H"]
        + predictions["p_draw"] * calibration["D"]
        + predictions["p_away_win"] * calibration["A"]
    )
    predictions["actual_spread"] = df["actual_spread"].values
    return predictions


def run_elo(df, k=3, home_field=20, spread_factor=55):
    """Run the Elo system over all matches and return results DataFrame."""
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    elo = Elo(k=k, home_field=home_field, spread_factor=spread_factor)

    pred_spreads = []
    pred_win_probs = []
    seasons = df["season"].unique()

    for season in seasons:
        season_mask = df["season"] == season
        season_games = df[season_mask]

        for idx, row in season_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            # Auto-add teams on first encounter
            if home not in elo.ratings:
                elo.add_team(home)
            if away not in elo.ratings:
                elo.add_team(away)

            # Record predictions before updating
            pred_spreads.append(elo.point_spread(home, away))
            pred_win_probs.append(elo.home_win_prob(home, away))

            # Update ratings with actual result
            elo.update_ratings(
                home, row["home_team_goals"],
                away, row["away_team_goals"],
            )

        # Regress between seasons
        elo.regress_towards_mean()

    df = df.copy()
    df["pred_spread"] = pred_spreads
    df["pred_home_win_prob"] = pred_win_probs

    return df, elo


def tune_parameters(df):
    """Grid search over Elo parameters to minimize MAE."""
    # Coarse grid
    k_values = [10, 15, 20, 25, 30, 35, 40]
    hf_values = [40, 50, 60, 70, 80, 90]
    sf_values = [28, 32, 36, 40, 44]

    best_mae = float("inf")
    best_params = {}

    total = len(k_values) * len(hf_values) * len(sf_values)
    print(f"Coarse grid search: {total} combinations")

    for k, hf, sf in product(k_values, hf_values, sf_values):
        results, _ = run_elo(df, k=k, home_field=hf, spread_factor=sf)
        mae = Elo.calculate_mae(results["pred_spread"], results["actual_spread"])
        if mae < best_mae:
            best_mae = mae
            best_params = {"k": k, "home_field": hf, "spread_factor": sf}

    print(f"Coarse best: MAE={best_mae:.4f}, params={best_params}")

    # Fine grid around best
    k_best = best_params["k"]
    hf_best = best_params["home_field"]
    sf_best = best_params["spread_factor"]

    k_fine = range(max(1, k_best - 8), k_best + 9, 1)
    hf_fine = range(max(1, hf_best - 20), hf_best + 21, 3)
    sf_fine = range(max(15, sf_best - 12), sf_best + 13, 1)

    total_fine = len(k_fine) * len(hf_fine) * len(sf_fine)
    print(f"Fine grid search: {total_fine} combinations")

    for k, hf, sf in product(k_fine, hf_fine, sf_fine):
        results, _ = run_elo(df, k=k, home_field=hf, spread_factor=sf)
        mae = Elo.calculate_mae(results["pred_spread"], results["actual_spread"])
        if mae < best_mae:
            best_mae = mae
            best_params = {"k": k, "home_field": hf, "spread_factor": sf}

    print(f"Fine best:   MAE={best_mae:.4f}, params={best_params}")
    return best_params, best_mae


def run_bayesian_elo(df, draws=2000, tune=2000, chains=4, target_accept=0.9):
    """Run the Bayesian rating system on match data.

    Parameters
    ----------
    df : DataFrame
        Match data from read_premier_results().
    draws, tune, chains, target_accept : int/float
        MCMC sampling parameters.

    Returns
    -------
    trace : arviz.InferenceData
    predictions : DataFrame
    data : dict
    """
    data = prepare_data(df)
    print(f"  Teams: {data['n_teams']}, Periods: {data['n_periods']}, "
          f"Matches: {len(data['outcome'])}")
    print(f"  Latent parameters: ~{data['n_teams'] * data['n_periods']}")

    model = build_model(data)
    print(f"  Sampling: {chains} chains x {draws} draws "
          f"(tune={tune}, target_accept={target_accept})")

    trace = fit_model(model, draws=draws, tune=tune, chains=chains,
                      target_accept=target_accept)

    predictions = predict_outcomes(trace, data)
    return trace, predictions, data


def predict_out_of_sample(trace, train_data, test_df):
    """Predict outcomes for held-out matches using trained Bayesian model.

    Uses the final-period posterior ratings from the training data to predict
    matches in test_df.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples from training.
    train_data : dict
        Output from prepare_data() on training data.
    test_df : DataFrame
        Held-out match data with home_team, away_team, result columns.

    Returns
    -------
    DataFrame with p_away_win, p_draw, p_home_win, pred_outcome, actual_outcome
    """
    team_to_idx = train_data["team_to_idx"]

    # Teams in test but not train get a sentinel index; we'll assign them
    # the cross-team mean rating (theta=0) for each posterior sample.
    all_test_teams = set(test_df["home_team"]) | set(test_df["away_team"])
    unseen = all_test_teams - set(team_to_idx)
    if unseen:
        print(f"  Unseen teams in test set (assigned mean rating): {sorted(unseen)}")

    # Sentinel index = n_teams (we'll append a zero-column for it)
    sentinel = train_data["n_teams"]
    home_idx = np.array([team_to_idx.get(t, sentinel) for t in test_df["home_team"]])
    away_idx = np.array([team_to_idx.get(t, sentinel) for t in test_df["away_team"]])

    theta_samples = trace.posterior["theta"].values
    n_chains, n_draws = theta_samples.shape[:2]
    theta_flat = theta_samples.reshape(n_chains * n_draws,
                                       train_data["n_periods"],
                                       train_data["n_teams"])

    # Use last training period for all test matches
    theta_last = theta_flat[:, -1, :]  # (n_samples, n_teams)

    # Append a zero-column for unseen teams (sentinel index)
    if unseen:
        zero_col = np.zeros((theta_last.shape[0], 1))
        theta_last = np.hstack([theta_last, zero_col])

    beta0 = trace.posterior["beta0"].values.flatten()
    beta1 = trace.posterior["beta1"].values.flatten()
    alpha0 = trace.posterior["alpha0"].values.flatten()
    alpha1 = trace.posterior["alpha1"].values.flatten()

    n_matches = len(home_idx)
    n_samples = len(beta0)

    batch_size = 200
    p_accum = np.zeros((n_matches, 3))

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_theta = theta_last[start:end]  # (batch, n_teams)
        batch_beta0 = beta0[start:end]
        batch_beta1 = beta1[start:end]
        batch_alpha0 = alpha0[start:end]
        batch_alpha1 = alpha1[start:end]

        th_home = batch_theta[:, home_idx]  # (batch, n_matches)
        th_away = batch_theta[:, away_idx]
        avg_th = (th_home + th_away) / 2

        x = 1.0
        b0 = batch_beta0[:, None]
        b1 = batch_beta1[:, None]
        a0 = batch_alpha0[:, None]
        a1 = batch_alpha1[:, None]

        log_p_hw = th_home + x * (a0 + a1 * avg_th) / 4
        log_p_aw = th_away - x * (a0 + a1 * avg_th) / 4
        log_p_d = b0 + (1 + b1) * avg_th

        logits = np.stack([log_p_aw, log_p_d, log_p_hw], axis=2)
        logits_max = logits.max(axis=2, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=2, keepdims=True)

        p_accum += probs.sum(axis=0)

    p_mean = p_accum / n_samples

    outcome_labels = {0: "A", 1: "D", 2: "H"}
    pred_outcome = [outcome_labels[i] for i in np.argmax(p_mean, axis=1)]
    actual_outcome = [r for r in test_df["result"]]

    return pd.DataFrame({
        "p_away_win": p_mean[:, 0],
        "p_draw": p_mean[:, 1],
        "p_home_win": p_mean[:, 2],
        "pred_outcome": pred_outcome,
        "actual_outcome": actual_outcome,
    })


class BayesianPredictor:
    """Stateful wrapper for incremental Bayesian rating prediction.

    Supports an initial fit on historical data followed by repeated
    predict → update cycles with lighter MCMC settings for weekly refits.
    """

    def __init__(self, historical_df,
                 initial_draws=1000, initial_tune=1000, initial_chains=4,
                 weekly_draws=500, weekly_tune=500, weekly_chains=2,
                 target_accept=0.9):
        self.df = historical_df.copy()
        self.initial_draws = initial_draws
        self.initial_tune = initial_tune
        self.initial_chains = initial_chains
        self.weekly_draws = weekly_draws
        self.weekly_tune = weekly_tune
        self.weekly_chains = weekly_chains
        self.target_accept = target_accept
        self.trace = None
        self.train_data = None

    def fit_initial(self):
        """Full MCMC fit on historical data (heavier settings)."""
        self.train_data = prepare_data(self.df)
        model = build_model(self.train_data)
        self.trace = fit_model(model,
                               draws=self.initial_draws,
                               tune=self.initial_tune,
                               chains=self.initial_chains,
                               target_accept=self.target_accept)

    def predict(self, games_df):
        """Predict upcoming games using current posterior.

        Parameters
        ----------
        games_df : DataFrame
            Games with home_team, away_team, result columns.

        Returns
        -------
        DataFrame with p_away_win, p_draw, p_home_win, pred_outcome,
        actual_outcome columns.
        """
        return predict_out_of_sample(self.trace, self.train_data, games_df)

    def update(self, completed_games_df):
        """Append results and refit with lighter MCMC settings."""
        self.df = pd.concat([self.df, completed_games_df], ignore_index=True)
        self.train_data = prepare_data(self.df)
        model = build_model(self.train_data)
        self.trace = fit_model(model,
                               draws=self.weekly_draws,
                               tune=self.weekly_tune,
                               chains=self.weekly_chains,
                               target_accept=self.target_accept)


def expanding_window_bayesian(df, draws=1000, tune=1000, chains=4,
                              target_accept=0.9):
    """Run expanding-window Bayesian predictions across all seasons.

    For each season s (from 2nd to last):
      - Train on seasons 1..s-1
      - Predict season s out of sample

    Returns a DataFrame with per-match Bayesian predictions for seasons 2..N,
    plus a 'season' column for later merging.
    """
    all_seasons = sorted(df["season"].unique())
    all_predictions = []

    for s_idx in range(1, len(all_seasons)):
        train_seasons = all_seasons[:s_idx]
        test_season = all_seasons[s_idx]

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy().reset_index(drop=True)

        print(f"  Window {s_idx}/{len(all_seasons)-1}: "
              f"train on {len(train_seasons)} seasons, "
              f"predict {test_season} ({len(test_df)} matches)")

        # Fit Bayesian model on training window
        train_data = prepare_data(train_df)
        model = build_model(train_data)
        trace = fit_model(model, draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept)

        # Predict held-out season
        bayes_preds = predict_out_of_sample(trace, train_data, test_df)

        # Compute spread calibration from training data only
        calibration = compute_spread_calibration(train_df)
        bayes_pred_spread = (
            bayes_preds["p_home_win"] * calibration["H"]
            + bayes_preds["p_draw"] * calibration["D"]
            + bayes_preds["p_away_win"] * calibration["A"]
        )

        # Collect per-match predictions
        window_df = pd.DataFrame({
            "season": test_season,
            "date": test_df["date"].values,
            "home_team": test_df["home_team"].values,
            "away_team": test_df["away_team"].values,
            "home_goals": test_df["home_team_goals"].values,
            "away_goals": test_df["away_team_goals"].values,
            "result": test_df["result"].values,
            "actual_spread": test_df["actual_spread"].values,
            "bayes_p_home_win": bayes_preds["p_home_win"].values,
            "bayes_p_draw": bayes_preds["p_draw"].values,
            "bayes_p_away_win": bayes_preds["p_away_win"].values,
            "bayes_pred_outcome": bayes_preds["pred_outcome"].values,
            "bayes_pred_spread": bayes_pred_spread.values,
        })
        all_predictions.append(window_df)

    return pd.concat(all_predictions, ignore_index=True)


def expanding_window_bayesian_weekly(df,
                                     initial_draws=1000, initial_tune=1000,
                                     initial_chains=4,
                                     weekly_draws=500, weekly_tune=500,
                                     weekly_chains=2,
                                     target_accept=0.9):
    """Run expanding-window Bayesian predictions with weekly in-season refits.

    For each test season s (from 2nd to last):
      1. Create BayesianPredictor on seasons 1..s-1, fit_initial()
      2. Group season s into weeks via group_games_by_week()
      3. For each week: predict() → collect predictions → update() with results
         (skip refit after the last week of each season)

    Returns a DataFrame with per-match Bayesian predictions for seasons 2..N,
    with the same schema as expanding_window_bayesian() plus a 'week' column.
    """
    all_seasons = sorted(df["season"].unique())
    all_predictions = []

    for s_idx in range(1, len(all_seasons)):
        train_seasons = all_seasons[:s_idx]
        test_season = all_seasons[s_idx]

        train_df = df[df["season"].isin(train_seasons)].copy()
        test_df = df[df["season"] == test_season].copy().reset_index(drop=True)

        weeks = group_games_by_week(test_df)
        n_weeks = len(weeks)

        print(f"  Season {s_idx}/{len(all_seasons)-1}: "
              f"train on {len(train_seasons)} seasons, "
              f"predict {test_season} ({len(test_df)} matches, "
              f"{n_weeks} weeks)")

        predictor = BayesianPredictor(
            train_df,
            initial_draws=initial_draws, initial_tune=initial_tune,
            initial_chains=initial_chains,
            weekly_draws=weekly_draws, weekly_tune=weekly_tune,
            weekly_chains=weekly_chains,
            target_accept=target_accept,
        )
        predictor.fit_initial()

        # Spread calibration from training data at start of season
        calibration = compute_spread_calibration(predictor.df)

        for w_idx, (week_label, week_df) in enumerate(weeks):
            week_df = week_df.reset_index(drop=True)
            bayes_preds = predictor.predict(week_df)

            bayes_pred_spread = (
                bayes_preds["p_home_win"] * calibration["H"]
                + bayes_preds["p_draw"] * calibration["D"]
                + bayes_preds["p_away_win"] * calibration["A"]
            )

            window_df = pd.DataFrame({
                "season": test_season,
                "week": week_label,
                "date": week_df["date"].values,
                "home_team": week_df["home_team"].values,
                "away_team": week_df["away_team"].values,
                "home_goals": week_df["home_team_goals"].values,
                "away_goals": week_df["away_team_goals"].values,
                "result": week_df["result"].values,
                "actual_spread": week_df["actual_spread"].values,
                "bayes_p_home_win": bayes_preds["p_home_win"].values,
                "bayes_p_draw": bayes_preds["p_draw"].values,
                "bayes_p_away_win": bayes_preds["p_away_win"].values,
                "bayes_pred_outcome": bayes_preds["pred_outcome"].values,
                "bayes_pred_spread": bayes_pred_spread.values,
            })
            all_predictions.append(window_df)

            # Skip refit after the last week of each season
            is_last_week = (w_idx == n_weeks - 1)
            if not is_last_week:
                # Update calibration with newly observed games
                calibration = compute_spread_calibration(
                    pd.concat([predictor.df, week_df], ignore_index=True)
                )
                predictor.update(week_df)

            print(f"    Week {w_idx+1}/{n_weeks} ({week_label}): "
                  f"{len(week_df)} games"
                  f"{'' if not is_last_week else ' [last, skip refit]'}")

    return pd.concat(all_predictions, ignore_index=True)


def run_classical_elo(df, params=None):
    """Run classical Elo and return metrics for comparison.

    Parameters
    ----------
    df : DataFrame
        Match data from read_premier_results().
    params : dict, optional
        Elo parameters (k, home_field, spread_factor). If None, uses
        TUNED_ELO_PARAMS.
    """
    if params is None:
        params = TUNED_ELO_PARAMS
    k = params["k"]
    hf = params["home_field"]
    sf = params["spread_factor"]

    results, elo = run_elo(df, k=k, home_field=hf, spread_factor=sf)

    # Win prediction accuracy (excl draws)
    results["pred_home_win"] = results["pred_home_win_prob"] > 0.5
    results["actual_home_win"] = results["actual_spread"] > 0
    non_draws = results[results["actual_spread"] != 0]
    win_accuracy = (
        non_draws["pred_home_win"] == non_draws["actual_home_win"]
    ).mean()

    mae = Elo.calculate_mae(results["pred_spread"], results["actual_spread"])

    rankings = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)

    return {
        "params": params,
        "mae": mae,
        "win_accuracy": win_accuracy,
        "rankings": rankings,
    }


def print_classical_results(classical):
    """Print classical Elo results."""
    print(f"  Parameters: {classical['params']}")
    print(f"  MAE (point spread): {classical['mae']:.4f}")
    print(f"  Win prediction accuracy (excl. draws): "
          f"{classical['win_accuracy']:.1%}")
    print()
    print("  Top 10 rankings:")
    for i, (team, rating) in enumerate(classical["rankings"][:10], 1):
        print(f"    {i:2d}. {team:25s} {rating:.1f}")


def print_bayesian_results(trace, predictions, data, df):
    """Print Bayesian Elo results and return metrics dict."""
    metrics = evaluate_predictions(predictions)
    print(f"  Categorical accuracy: {metrics['categorical_accuracy']:.1%}")
    print(f"  Log-loss: {metrics['log_loss']:.4f}")
    print(f"  Decisive game accuracy: {metrics['decisive_accuracy']:.1%}")
    print()

    draw_rate_actual = (predictions["actual_outcome"] == "D").mean()
    draw_rate_pred = predictions["p_draw"].mean()
    print(f"  Actual draw rate: {draw_rate_actual:.1%}")
    print(f"  Mean predicted P(draw): {draw_rate_pred:.1%}")
    print()

    calibration = compute_spread_calibration(df)
    predictions = bayesian_spread_predictions(predictions, calibration, df)
    spread_mae = (predictions["pred_spread"] - predictions["actual_spread"]).abs().mean()
    metrics["spread_mae"] = spread_mae

    print(f"  Spread calibration: E[spread|H]={calibration['H']:.2f}, "
          f"E[spread|D]={calibration['D']:.2f}, "
          f"E[spread|A]={calibration['A']:.2f}")
    print(f"  Point spread MAE (calibrated): {spread_mae:.4f}")
    print()

    ratings_df = extract_ratings(trace, data, period=-1)
    print("  Top 10 rankings (final season):")
    for i, row in ratings_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['team']:25s} "
              f"{row['rating_mean']:.0f} "
              f"[{row['ci_lower']:.0f}, {row['ci_upper']:.0f}]")

    return metrics


def print_comparison(classical, metrics):
    """Print side-by-side comparison table."""
    print(f"{'Metric':<35s} {'Classical':>12s} {'Bayesian':>12s}")
    print("-" * 60)
    print(f"{'Win accuracy (excl. draws)':<35s} "
          f"{classical['win_accuracy']:>11.1%} "
          f"{metrics['decisive_accuracy']:>11.1%}")
    print(f"{'Categorical accuracy (H/D/A)':<35s} "
          f"{'N/A':>12s} "
          f"{metrics['categorical_accuracy']:>11.1%}")
    print(f"{'Log-loss':<35s} "
          f"{'N/A':>12s} "
          f"{metrics['log_loss']:>12.4f}")
    print(f"{'Point spread MAE':<35s} "
          f"{classical['mae']:>12.4f} "
          f"{metrics['spread_mae']:>12.4f}")


def generate_comparison_csv(df, output_path="data/premier-league-sequential-compare.csv"):
    """Expanding-window sequential comparison of Elo vs Bayesian.

    Classical Elo runs sequentially over all data (online updates).
    Bayesian model retrains on expanding windows (seasons 1..s-1) and
    predicts season s, for s = 2..N.

    Returns
    -------
    tuple of (csv_df, metrics_dict)
    """
    all_seasons = sorted(df["season"].unique())
    n_seasons = len(all_seasons)
    bayes_seasons = all_seasons[1:]  # seasons 2..N (first is training-only)

    # --- Classical Elo (run on ALL data sequentially) ---
    print("Running Classical Elo on all data (sequential)...")
    params = TUNED_ELO_PARAMS
    elo_results, _ = run_elo(df, k=params["k"], home_field=params["home_field"],
                             spread_factor=params["spread_factor"])
    # Filter to seasons 2..N to match Bayesian coverage
    elo_results = elo_results[
        elo_results["season"].isin(bayes_seasons)
    ].reset_index(drop=True)
    print(f"  Elo predictions: {len(elo_results)} matches "
          f"across {len(bayes_seasons)} seasons")
    print()

    # --- Bayesian expanding-window ---
    print(f"Running Bayesian expanding-window ({n_seasons - 1} MCMC fits)...")
    bayes_results = expanding_window_bayesian(df)
    print(f"  Bayesian predictions: {len(bayes_results)} matches")
    print()

    # --- Merge into per-match CSV ---
    csv_df = bayes_results.copy()
    csv_df["elo_pred_spread"] = elo_results["pred_spread"].values
    csv_df["elo_pred_home_win_prob"] = elo_results["pred_home_win_prob"].values

    csv_df.to_csv(output_path, index=False)
    print(f"Saved {len(csv_df)} rows to {output_path}")

    # --- Compute summary metrics ---
    # Elo metrics
    elo_mae = Elo.calculate_mae(csv_df["elo_pred_spread"], csv_df["actual_spread"])
    elo_rmse = Elo.calculate_rmse(csv_df["elo_pred_spread"], csv_df["actual_spread"])
    elo_non_draws = csv_df[csv_df["result"] != "D"]
    elo_pred_home_win = elo_non_draws["elo_pred_home_win_prob"] > 0.5
    elo_actual_home_win = elo_non_draws["actual_spread"] > 0
    elo_win_acc = (elo_pred_home_win == elo_actual_home_win).mean()

    # Bayesian metrics
    bayes_mae = Elo.calculate_mae(csv_df["bayes_pred_spread"], csv_df["actual_spread"])
    bayes_rmse = Elo.calculate_rmse(csv_df["bayes_pred_spread"], csv_df["actual_spread"])
    bayes_cat_acc = (csv_df["bayes_pred_outcome"] == csv_df["result"]).mean()
    bayes_decisive = csv_df[csv_df["result"] != "D"]
    bayes_decisive_correct = (
        bayes_decisive["bayes_pred_outcome"] == bayes_decisive["result"]
    )
    bayes_win_acc = bayes_decisive_correct.mean()

    # Bayesian log-loss
    eps = 1e-10
    log_loss_vals = []
    for _, row in csv_df.iterrows():
        if row["result"] == "A":
            p = row["bayes_p_away_win"]
        elif row["result"] == "D":
            p = row["bayes_p_draw"]
        else:
            p = row["bayes_p_home_win"]
        log_loss_vals.append(-np.log(np.clip(p, eps, 1.0)))
    bayes_log_loss = np.mean(log_loss_vals)

    # Overall spread calibration (from all training data for summary)
    calibration = compute_spread_calibration(df)

    metrics = {
        "n_matches": len(csv_df),
        "n_seasons": len(bayes_seasons),
        "first_test_season": bayes_seasons[0],
        "last_test_season": bayes_seasons[-1],
        "calibration": calibration,
        "elo_params": params,
        "elo_mae": elo_mae,
        "elo_rmse": elo_rmse,
        "elo_win_acc": elo_win_acc,
        "bayes_mae": bayes_mae,
        "bayes_rmse": bayes_rmse,
        "bayes_cat_acc": bayes_cat_acc,
        "bayes_win_acc": bayes_win_acc,
        "bayes_log_loss": bayes_log_loss,
    }

    return csv_df, metrics


def elo_3way_probs(elo_home_win_prob, draw_rate):
    """Convert Elo 2-way home-win probability to 3-way (H/D/A) probabilities.

    Uses a fixed empirical draw rate to carve out draw probability, then
    splits the remaining mass using the Elo-derived home-win probability.

    Parameters
    ----------
    elo_home_win_prob : array-like
        Elo-derived P(home win) for each match (2-way, no draws).
    draw_rate : float
        Empirical draw rate from training data.

    Returns
    -------
    tuple of (p_home_win, p_draw, p_away_win), each as np.ndarray
    """
    elo_home_win_prob = np.asarray(elo_home_win_prob)
    p_draw = np.full_like(elo_home_win_prob, draw_rate)
    p_home = (1 - draw_rate) * elo_home_win_prob
    p_away = (1 - draw_rate) * (1 - elo_home_win_prob)
    return p_home, p_draw, p_away


def cross_entropy_evaluation(df,
                             output_path="data/cross-entropy-evaluation.csv",
                             weekly=False,
                             initial_draws=1000, initial_tune=1000,
                             initial_chains=4,
                             weekly_draws=500, weekly_tune=500,
                             weekly_chains=2):
    """Expanding-window cross-entropy comparison: Bayesian vs Elo vs baseline.

    For each test season s (from 2nd to last):
      - Bayesian: train on 1..s-1, predict s (direct 3-way probabilities)
      - Elo: sequential online updates, convert to 3-way using per-window
        empirical draw rate
      - Baseline: empirical H/D/A frequencies from training seasons

    When weekly=True, uses weekly in-season refits instead of per-season
    predictions for the Bayesian model.

    Returns
    -------
    tuple of (per_game_df, summary_df)
        per_game_df has per-match probabilities and cross-entropy contributions.
        summary_df has per-season and aggregate cross-entropy for each model.
    """
    all_seasons = sorted(df["season"].unique())
    n_windows = len(all_seasons) - 1

    # --- Run Elo on ALL data (sequential online updates) ---
    print("Running Classical Elo on all data (sequential)...")
    params = TUNED_ELO_PARAMS
    elo_results, _ = run_elo(df, k=params["k"], home_field=params["home_field"],
                             spread_factor=params["spread_factor"])
    print()

    # --- Run Bayesian expanding-window ---
    if weekly:
        print(f"Running Bayesian weekly expanding-window "
              f"({n_windows} seasons, weekly refits)...")
        print(f"  Initial MCMC: {initial_chains} chains x {initial_draws} "
              f"draws (tune={initial_tune})")
        print(f"  Weekly MCMC:  {weekly_chains} chains x {weekly_draws} "
              f"draws (tune={weekly_tune})")
        bayes_results = expanding_window_bayesian_weekly(
            df,
            initial_draws=initial_draws, initial_tune=initial_tune,
            initial_chains=initial_chains,
            weekly_draws=weekly_draws, weekly_tune=weekly_tune,
            weekly_chains=weekly_chains,
        )
    else:
        print(f"Running Bayesian expanding-window ({n_windows} MCMC fits)...")
        bayes_results = expanding_window_bayesian(df)
    print()

    # --- Build per-game evaluation DataFrame ---
    # Align Elo results with Bayesian results (seasons 2..N)
    bayes_seasons = all_seasons[1:]
    elo_test = elo_results[
        elo_results["season"].isin(bayes_seasons)
    ].reset_index(drop=True)

    season_dfs = []
    for s_idx in range(1, len(all_seasons)):
        test_season = all_seasons[s_idx]
        train_seasons = all_seasons[:s_idx]
        train_df = df[df["season"].isin(train_seasons)]

        # Empirical outcome frequencies from training data
        train_counts = train_df["result"].value_counts(normalize=True)
        freq_h = train_counts.get("H", 0.0)
        freq_d = train_counts.get("D", 0.0)
        freq_a = train_counts.get("A", 0.0)

        # Season slices
        bayes_season = bayes_results[
            bayes_results["season"] == test_season
        ].reset_index(drop=True)
        elo_season = elo_test[
            elo_test["season"] == test_season
        ].reset_index(drop=True)

        n = len(bayes_season)

        # Elo 3-way probs using training draw rate
        elo_p_home, elo_p_draw, elo_p_away = elo_3way_probs(
            elo_season["pred_home_win_prob"].values, freq_d
        )

        result = bayes_season["result"].values

        # Bayesian P(actual outcome)
        bayes_p_actual = np.where(
            result == "H", bayes_season["bayes_p_home_win"].values,
            np.where(result == "D", bayes_season["bayes_p_draw"].values,
                     bayes_season["bayes_p_away_win"].values))

        # Elo P(actual outcome)
        elo_p_actual = np.where(
            result == "H", elo_p_home,
            np.where(result == "D", elo_p_draw, elo_p_away))

        # Baseline P(actual outcome)
        baseline_p_actual = np.where(
            result == "H", freq_h,
            np.where(result == "D", freq_d, freq_a))

        eps = 1e-10
        s_df_data = {
            "season": test_season,
            "date": bayes_season["date"].values,
            "home_team": bayes_season["home_team"].values,
            "away_team": bayes_season["away_team"].values,
            "result": result,
            "bayes_p_home": bayes_season["bayes_p_home_win"].values,
            "bayes_p_draw": bayes_season["bayes_p_draw"].values,
            "bayes_p_away": bayes_season["bayes_p_away_win"].values,
            "elo_p_home": elo_p_home,
            "elo_p_draw": elo_p_draw,
            "elo_p_away": elo_p_away,
            "baseline_p_home": freq_h,
            "baseline_p_draw": freq_d,
            "baseline_p_away": freq_a,
            "bayes_ce": -np.log(np.clip(bayes_p_actual, eps, 1.0)),
            "elo_ce": -np.log(np.clip(elo_p_actual, eps, 1.0)),
            "baseline_ce": -np.log(np.clip(baseline_p_actual, eps, 1.0)),
        }
        if weekly and "week" in bayes_season.columns:
            s_df_data["week"] = bayes_season["week"].values
        s_df = pd.DataFrame(s_df_data)
        season_dfs.append(s_df)

    per_game_df = pd.concat(season_dfs, ignore_index=True)
    per_game_df.to_csv(output_path, index=False)
    print(f"Saved {len(per_game_df)} per-game rows to {output_path}")

    # --- Compute per-season and aggregate summary ---
    summary_rows = []
    for season in bayes_seasons:
        s_df = per_game_df[per_game_df["season"] == season]
        n = len(s_df)
        summary_rows.append({
            "season": season,
            "n_matches": n,
            "bayes_ce": s_df["bayes_ce"].mean(),
            "elo_ce": s_df["elo_ce"].mean(),
            "baseline_ce": s_df["baseline_ce"].mean(),
        })

    # Aggregate row
    summary_rows.append({
        "season": "ALL",
        "n_matches": len(per_game_df),
        "bayes_ce": per_game_df["bayes_ce"].mean(),
        "elo_ce": per_game_df["elo_ce"].mean(),
        "baseline_ce": per_game_df["baseline_ce"].mean(),
    })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = output_path.replace(".csv", "-summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    return per_game_df, summary_df


def training_depth_experiment(df, test_season="2021-22",
                              depths=None,
                              weekly=True,
                              initial_draws=1000, initial_tune=1000,
                              initial_chains=4,
                              weekly_draws=500, weekly_tune=500,
                              weekly_chains=2):
    """Compare Bayesian/Elo/baseline across different training window depths.

    Holds out test_season and varies how many prior seasons are used for
    training.  Each depth runs Elo (sequential online), a naive baseline
    (empirical H/D/A frequencies), and the Bayesian model on the same
    training slice.

    Parameters
    ----------
    df : DataFrame
        Full match data from read_premier_results().
    test_season : str
        Season to hold out as the test set.
    depths : list
        Training-window specifications. Each element is one of:
        - "all"  — all prior seasons
        - "half" — floor(n_prior / 2) most recent prior seasons
        - int    — that many most recent prior seasons
        Defaults to ["all", "half", 1].
    weekly : bool
        If True, use BayesianPredictor with weekly in-season refits.
        If False, use a single predict_out_of_sample call.
    initial_draws, initial_tune, initial_chains : int
        MCMC settings for the initial Bayesian fit.
    weekly_draws, weekly_tune, weekly_chains : int
        MCMC settings for weekly Bayesian refits (only used when weekly=True).

    Returns
    -------
    (per_game_df, summary_df)
    """
    if depths is None:
        depths = ["all", "half", 1]

    all_seasons = sorted(df["season"].unique())
    if test_season not in all_seasons:
        raise ValueError(f"test_season {test_season!r} not found in data")

    test_idx = all_seasons.index(test_season)
    prior_seasons = all_seasons[:test_idx]
    n_prior = len(prior_seasons)
    if n_prior == 0:
        raise ValueError(f"No training seasons before {test_season}")

    test_df = df[df["season"] == test_season].copy().reset_index(drop=True)
    eps = 1e-10

    all_per_game = []

    for depth_spec in depths:
        # Determine which training seasons to use
        if depth_spec == "all":
            n_train = n_prior
        elif depth_spec == "half":
            n_train = n_prior // 2
        elif isinstance(depth_spec, int):
            n_train = min(depth_spec, n_prior)
        else:
            raise ValueError(f"Invalid depth spec: {depth_spec!r}")

        train_seasons = prior_seasons[-n_train:]
        depth_label = (f"all ({n_train} seasons)" if depth_spec == "all"
                       else f"half ({n_train} seasons)" if depth_spec == "half"
                       else f"{n_train} season{'s' if n_train != 1 else ''}")
        train_df = df[df["season"].isin(train_seasons)].copy()

        print(f"\n{'=' * 60}")
        print(f"Depth: {depth_label}")
        print(f"  Training seasons: {train_seasons[0]} .. {train_seasons[-1]}")
        print(f"  Training matches: {len(train_df)}")
        print(f"  Test season: {test_season} ({len(test_df)} matches)")
        print("=" * 60)

        # --- Baseline: empirical H/D/A frequencies ---
        freq = train_df["result"].value_counts(normalize=True)
        freq_h = freq.get("H", 0.0)
        freq_d = freq.get("D", 0.0)
        freq_a = freq.get("A", 0.0)

        # --- Elo on training slice + test season ---
        print("\n  Running Elo...")
        elo_df = pd.concat([train_df, test_df], ignore_index=True)
        params = TUNED_ELO_PARAMS
        elo_results, _ = run_elo(elo_df, k=params["k"],
                                 home_field=params["home_field"],
                                 spread_factor=params["spread_factor"])
        elo_test = elo_results[
            elo_results["season"] == test_season
        ].reset_index(drop=True)
        elo_p_home, elo_p_draw, elo_p_away = elo_3way_probs(
            elo_test["pred_home_win_prob"].values, freq_d
        )

        # --- Bayesian ---
        if weekly:
            print("  Running Bayesian (weekly refits)...")
            predictor = BayesianPredictor(
                train_df,
                initial_draws=initial_draws, initial_tune=initial_tune,
                initial_chains=initial_chains,
                weekly_draws=weekly_draws, weekly_tune=weekly_tune,
                weekly_chains=weekly_chains,
            )
            predictor.fit_initial()

            weeks = group_games_by_week(test_df)
            n_weeks = len(weeks)
            bayes_parts = []

            for w_idx, (week_label, week_df) in enumerate(weeks):
                week_df = week_df.reset_index(drop=True)
                preds = predictor.predict(week_df)
                preds["week"] = week_label
                preds["date"] = week_df["date"].values
                preds["home_team"] = week_df["home_team"].values
                preds["away_team"] = week_df["away_team"].values
                preds["result"] = week_df["result"].values
                bayes_parts.append(preds)

                is_last = (w_idx == n_weeks - 1)
                if not is_last:
                    predictor.update(week_df)
                print(f"    Week {w_idx+1}/{n_weeks} ({week_label}): "
                      f"{len(week_df)} games"
                      f"{'' if not is_last else ' [last, skip refit]'}")

            bayes_preds = pd.concat(bayes_parts, ignore_index=True)
        else:
            print("  Running Bayesian (per-season)...")
            train_data = prepare_data(train_df)
            model = build_model(train_data)
            trace = fit_model(model, draws=initial_draws, tune=initial_tune,
                              chains=initial_chains, target_accept=0.9)
            bayes_preds = predict_out_of_sample(trace, train_data, test_df)
            bayes_preds["date"] = test_df["date"].values
            bayes_preds["home_team"] = test_df["home_team"].values
            bayes_preds["away_team"] = test_df["away_team"].values
            bayes_preds["result"] = test_df["result"].values

        # --- Assemble per-game results ---
        result = bayes_preds["result"].values

        bayes_p_actual = np.where(
            result == "H", bayes_preds["p_home_win"].values,
            np.where(result == "D", bayes_preds["p_draw"].values,
                     bayes_preds["p_away_win"].values))
        elo_p_actual = np.where(
            result == "H", elo_p_home,
            np.where(result == "D", elo_p_draw, elo_p_away))
        baseline_p_actual = np.where(
            result == "H", freq_h,
            np.where(result == "D", freq_d, freq_a))

        row_data = {
            "depth_label": depth_label,
            "season": test_season,
            "date": bayes_preds["date"].values,
            "home_team": bayes_preds["home_team"].values,
            "away_team": bayes_preds["away_team"].values,
            "result": result,
            "bayes_p_home": bayes_preds["p_home_win"].values,
            "bayes_p_draw": bayes_preds["p_draw"].values,
            "bayes_p_away": bayes_preds["p_away_win"].values,
            "elo_p_home": elo_p_home,
            "elo_p_draw": elo_p_draw,
            "elo_p_away": elo_p_away,
            "baseline_p_home": freq_h,
            "baseline_p_draw": freq_d,
            "baseline_p_away": freq_a,
            "bayes_ce": -np.log(np.clip(bayes_p_actual, eps, 1.0)),
            "elo_ce": -np.log(np.clip(elo_p_actual, eps, 1.0)),
            "baseline_ce": -np.log(np.clip(baseline_p_actual, eps, 1.0)),
        }
        if weekly and "week" in bayes_preds.columns:
            row_data["week"] = bayes_preds["week"].values

        depth_df = pd.DataFrame(row_data)
        all_per_game.append((depth_df, n_train, len(train_df)))

    per_game_df = pd.concat([t[0] for t in all_per_game], ignore_index=True)

    # --- Summary: one row per depth ---
    summary_rows = []
    for depth_df, n_train_seasons, n_train_matches in all_per_game:
        dl = depth_df["depth_label"].iloc[0]
        summary_rows.append({
            "depth_label": dl,
            "n_train_seasons": n_train_seasons,
            "n_train_matches": n_train_matches,
            "n_test_matches": len(depth_df),
            "bayes_ce": depth_df["bayes_ce"].mean(),
            "elo_ce": depth_df["elo_ce"].mean(),
            "baseline_ce": depth_df["baseline_ce"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    return per_game_df, summary_df


def print_training_depth_results(summary_df, test_season="2021-22",
                                 weekly=False):
    """Print training depth experiment results."""
    mode_label = ("weekly in-season refits" if weekly
                  else "per-season predictions")
    print()
    print("=" * 70)
    print(f"TRAINING DEPTH EXPERIMENT — Test season: {test_season}")
    print(f"  Bayesian mode: {mode_label}")
    print("=" * 70)
    print()

    print(f"  {'Depth':<22s} {'Train':>7s} {'Baseline':>10s} "
          f"{'Elo':>10s} {'Bayesian':>10s} {'Bayes-Elo':>10s}")
    print("  " + "-" * 70)
    for _, row in summary_df.iterrows():
        diff = row["bayes_ce"] - row["elo_ce"]
        print(f"  {row['depth_label']:<22s} {int(row['n_train_matches']):>7d} "
              f"{row['baseline_ce']:>10.4f} {row['elo_ce']:>10.4f} "
              f"{row['bayes_ce']:>10.4f} {diff:>+10.4f}")

    print()
    best = summary_df.loc[summary_df["bayes_ce"].idxmin()]
    print(f"  Lowest Bayesian CE: {best['depth_label']} "
          f"({best['bayes_ce']:.4f})")


def print_cross_entropy_results(summary_df, weekly=False):
    """Print cross-entropy evaluation results."""
    agg = summary_df[summary_df["season"] == "ALL"].iloc[0]
    seasons = summary_df[summary_df["season"] != "ALL"]

    mode_label = "weekly in-season refits" if weekly else "per-season predictions"

    print()
    print("=" * 70)
    print("CROSS-ENTROPY EVALUATION (a la Glickman 2025, Section 6)")
    print(f"  Bayesian mode: {mode_label}")
    print("=" * 70)
    print()
    print(f"Total matches evaluated: {int(agg['n_matches'])}")
    print()

    # Aggregate results
    print("Aggregate Cross-Entropy (lower is better):")
    print(f"  {'Model':<20s} {'Cross-Entropy':>14s} {'vs Baseline':>14s}")
    print("  " + "-" * 50)
    print(f"  {'Naive baseline':<20s} {agg['baseline_ce']:>14.4f} {'--':>14s}")
    print(f"  {'Classical Elo':<20s} {agg['elo_ce']:>14.4f} "
          f"{(1 - agg['elo_ce'] / agg['baseline_ce']):>13.1%}")
    print(f"  {'Bayesian':<20s} {agg['bayes_ce']:>14.4f} "
          f"{(1 - agg['bayes_ce'] / agg['baseline_ce']):>13.1%}")
    print()

    # Per-season breakdown
    print("Per-Season Cross-Entropy:")
    print(f"  {'Season':<12s} {'N':>5s} {'Baseline':>10s} "
          f"{'Elo':>10s} {'Bayesian':>10s} {'Bayes-Elo':>10s}")
    print("  " + "-" * 60)
    for _, row in seasons.iterrows():
        diff = row["bayes_ce"] - row["elo_ce"]
        print(f"  {str(row['season']):<12s} {int(row['n_matches']):>5d} "
              f"{row['baseline_ce']:>10.4f} {row['elo_ce']:>10.4f} "
              f"{row['bayes_ce']:>10.4f} {diff:>+10.4f}")

    print()
    bayes_better = (seasons["bayes_ce"] < seasons["elo_ce"]).sum()
    total = len(seasons)
    print(f"Bayesian lower CE than Elo in {bayes_better}/{total} seasons")


def generate_analysis_markdown(metrics,
                               output_path="data/sequential-comparison-analysis.md"):
    """Write a Markdown analysis of the expanding-window comparison."""
    cal = metrics["calibration"]
    lines = [
        "# Expanding-Window Sequential Comparison: Classical Elo vs Bayesian Elo",
        "",
        "## Methodology",
        "",
        f"- **Data:** {metrics['n_seasons'] + 1} Premier League seasons "
        f"({metrics['first_test_season']} through {metrics['last_test_season']} "
        f"used as test seasons)",
        f"- **Evaluation matches:** {metrics['n_matches']} "
        f"(all matches from seasons 2 through {metrics['n_seasons'] + 1})",
        "- **Classical Elo:** Run sequentially over all seasons; each prediction "
        "is made before observing that match's result (online updates). "
        f"Parameters: K={metrics['elo_params']['k']}, "
        f"home_field={metrics['elo_params']['home_field']}, "
        f"spread_factor={metrics['elo_params']['spread_factor']}.",
        "- **Bayesian Elo:** Expanding-window retraining. For each test season s, "
        "a full MCMC fit is run on all prior seasons (1..s-1). Predictions for "
        "season s use the final-period posterior ratings from that fit. "
        "This gives the Bayesian model access to all historical data while "
        "ensuring genuinely forward-looking predictions.",
        "- **Spread calibration:** Computed per-window from training data only. "
        f"Overall calibration: "
        f"E[spread|H]={cal['H']:.2f}, "
        f"E[spread|D]={cal['D']:.2f}, "
        f"E[spread|A]={cal['A']:.2f}.",
        "",
        "## Outcome Prediction Accuracy",
        "",
        f"- **Bayesian categorical accuracy (H/D/A):** "
        f"{metrics['bayes_cat_acc']:.1%}",
        f"- **Bayesian log-loss:** {metrics['bayes_log_loss']:.4f}",
        f"- **Win/loss accuracy (excl. draws):** "
        f"Classical Elo {metrics['elo_win_acc']:.1%}, "
        f"Bayesian {metrics['bayes_win_acc']:.1%}",
        "",
        "## Point Spread Prediction",
        "",
        f"| Metric | Classical Elo | Bayesian Elo |",
        f"|--------|--------------|-------------|",
        f"| MAE    | {metrics['elo_mae']:.4f} | {metrics['bayes_mae']:.4f} |",
        f"| RMSE   | {metrics['elo_rmse']:.4f} | {metrics['bayes_rmse']:.4f} |",
        "",
        "## Summary Comparison",
        "",
        f"| Metric | Classical Elo | Bayesian Elo |",
        f"|--------|--------------|-------------|",
        f"| Win accuracy (excl. draws) | {metrics['elo_win_acc']:.1%} "
        f"| {metrics['bayes_win_acc']:.1%} |",
        f"| Categorical accuracy (H/D/A) | N/A "
        f"| {metrics['bayes_cat_acc']:.1%} |",
        f"| Log-loss | N/A | {metrics['bayes_log_loss']:.4f} |",
        f"| Spread MAE | {metrics['elo_mae']:.4f} "
        f"| {metrics['bayes_mae']:.4f} |",
        f"| Spread RMSE | {metrics['elo_rmse']:.4f} "
        f"| {metrics['bayes_rmse']:.4f} |",
        "",
        "## Observations",
        "",
    ]

    # Generate observations based on metric comparisons
    if metrics['bayes_mae'] < metrics['elo_mae']:
        mae_diff = metrics['elo_mae'] - metrics['bayes_mae']
        lines.append(
            f"- Bayesian Elo achieves a lower spread MAE by {mae_diff:.4f}, "
            f"suggesting better calibrated point spread predictions."
        )
    elif metrics['elo_mae'] < metrics['bayes_mae']:
        mae_diff = metrics['bayes_mae'] - metrics['elo_mae']
        lines.append(
            f"- Classical Elo achieves a lower spread MAE by {mae_diff:.4f}, "
            f"suggesting better calibrated point spread predictions."
        )
    else:
        lines.append("- Both models achieve identical spread MAE.")

    if metrics['bayes_win_acc'] > metrics['elo_win_acc']:
        lines.append(
            "- Bayesian model shows higher win/loss prediction accuracy on decisive games."
        )
    elif metrics['elo_win_acc'] > metrics['bayes_win_acc']:
        lines.append(
            "- Classical Elo shows higher win/loss prediction accuracy on decisive games."
        )

    lines.append(
        "- The Bayesian model provides full three-way probability estimates (H/D/A), "
        "enabling categorical predictions and proper log-loss evaluation."
    )
    lines.append(
        "- Both models update over time: Classical Elo updates per-match (online), "
        "while Bayesian Elo retrains per-season on expanding windows of data. "
        "This is a fairer comparison than a single held-out season."
    )
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Analysis written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Premier League rating system comparison"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Elo parameter grid search instead of using saved best params"
    )
    parser.add_argument(
        "--elo-only", action="store_true",
        help="Run only the classical Elo system"
    )
    parser.add_argument(
        "--bayesian-only", action="store_true",
        help="Run only the Bayesian system"
    )
    parser.add_argument(
        "--compare-2022", action="store_true",
        help="Run expanding-window sequential comparison (Elo vs Bayesian)"
    )
    parser.add_argument(
        "--cross-entropy", action="store_true",
        help="Run cross-entropy evaluation (Bayesian vs Elo vs baseline)"
    )
    parser.add_argument(
        "--depth-experiment", action="store_true",
        help="Run training depth experiment on 2021-22"
    )
    parser.add_argument(
        "--weekly", action="store_true",
        help="Use weekly in-season Bayesian refits (with --cross-entropy "
             "or --depth-experiment)"
    )
    parser.add_argument(
        "--initial-draws", type=int, default=1000,
        help="MCMC draws for initial fit (default: 1000)"
    )
    parser.add_argument(
        "--initial-tune", type=int, default=1000,
        help="MCMC tune steps for initial fit (default: 1000)"
    )
    parser.add_argument(
        "--initial-chains", type=int, default=4,
        help="MCMC chains for initial fit (default: 4)"
    )
    parser.add_argument(
        "--weekly-draws", type=int, default=500,
        help="MCMC draws for weekly refits (default: 500)"
    )
    parser.add_argument(
        "--weekly-tune", type=int, default=500,
        help="MCMC tune steps for weekly refits (default: 500)"
    )
    parser.add_argument(
        "--weekly-chains", type=int, default=2,
        help="MCMC chains for weekly refits (default: 2)"
    )
    args = parser.parse_args()

    df = read_premier_results()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons")
    print()

    if args.cross_entropy:
        _, summary_df = cross_entropy_evaluation(
            df,
            weekly=args.weekly,
            initial_draws=args.initial_draws,
            initial_tune=args.initial_tune,
            initial_chains=args.initial_chains,
            weekly_draws=args.weekly_draws,
            weekly_tune=args.weekly_tune,
            weekly_chains=args.weekly_chains,
        )
        print_cross_entropy_results(summary_df, weekly=args.weekly)
        return

    if args.depth_experiment:
        per_game_df, summary_df = training_depth_experiment(
            df,
            test_season="2021-22",
            weekly=args.weekly,
            initial_draws=args.initial_draws,
            initial_tune=args.initial_tune,
            initial_chains=args.initial_chains,
            weekly_draws=args.weekly_draws,
            weekly_tune=args.weekly_tune,
            weekly_chains=args.weekly_chains,
        )
        per_game_df.to_csv("data/depth-experiment.csv", index=False)
        summary_df.to_csv("data/depth-experiment-summary.csv", index=False)
        print(f"\nSaved {len(per_game_df)} per-game rows to "
              f"data/depth-experiment.csv")
        print_training_depth_results(summary_df, test_season="2021-22",
                                     weekly=args.weekly)
        return

    if args.compare_2022:
        print("=" * 60)
        print("EXPANDING-WINDOW SEQUENTIAL COMPARISON")
        print("=" * 60)
        _, metrics = generate_comparison_csv(df)
        print()
        generate_analysis_markdown(metrics)
        return

    run_elo_flag = not args.bayesian_only
    run_bayesian_flag = not args.elo_only

    classical = None
    metrics = None

    if run_elo_flag:
        print("=" * 60)
        print("CLASSICAL ELO")
        print("=" * 60)
        if args.tune:
            print("Tuning parameters...")
            best_params, _ = tune_parameters(df)
            print()
        else:
            best_params = TUNED_ELO_PARAMS
        classical = run_classical_elo(df, params=best_params)
        print_classical_results(classical)
        print()

    if run_bayesian_flag:
        print("=" * 60)
        print("BAYESIAN ELO (PyMC)")
        print("=" * 60)
        trace, predictions, data = run_bayesian_elo(df)
        print()
        metrics = print_bayesian_results(trace, predictions, data, df)
        print()

    if classical and metrics:
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print_comparison(classical, metrics)


if __name__ == "__main__":
    main()
