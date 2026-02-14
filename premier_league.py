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


def run_classical_elo(df):
    """Run classical Elo and return metrics for comparison."""
    best_params, best_mae = tune_parameters(df)
    k = best_params["k"]
    hf = best_params["home_field"]
    sf = best_params["spread_factor"]

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
        "params": best_params,
        "mae": mae,
        "win_accuracy": win_accuracy,
        "rankings": rankings,
    }


def main():
    df = read_premier_results()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons")
    print()

    # --- Classical Elo ---
    print("=" * 60)
    print("CLASSICAL ELO")
    print("=" * 60)
    classical = run_classical_elo(df)
    print(f"  Parameters: {classical['params']}")
    print(f"  MAE (point spread): {classical['mae']:.4f}")
    print(f"  Win prediction accuracy (excl. draws): "
          f"{classical['win_accuracy']:.1%}")
    print()
    print("  Top 10 rankings:")
    for i, (team, rating) in enumerate(classical["rankings"][:10], 1):
        print(f"    {i:2d}. {team:25s} {rating:.1f}")
    print()

    # --- Bayesian Elo ---
    print("=" * 60)
    print("BAYESIAN ELO (PyMC)")
    print("=" * 60)
    trace, predictions, data = run_bayesian_elo(df)
    print()

    # Evaluation
    metrics = evaluate_predictions(predictions)
    print(f"  Categorical accuracy: {metrics['categorical_accuracy']:.1%}")
    print(f"  Log-loss: {metrics['log_loss']:.4f}")
    print(f"  Decisive game accuracy: {metrics['decisive_accuracy']:.1%}")
    print()

    # Draw calibration
    draw_rate_actual = (predictions["actual_outcome"] == "D").mean()
    draw_rate_pred = predictions["p_draw"].mean()
    print(f"  Actual draw rate: {draw_rate_actual:.1%}")
    print(f"  Mean predicted P(draw): {draw_rate_pred:.1%}")
    print()

    # Rankings with uncertainty
    ratings_df = extract_ratings(trace, data, period=-1)
    print("  Top 10 rankings (final season):")
    for i, row in ratings_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['team']:25s} "
              f"{row['rating_mean']:.0f} "
              f"[{row['ci_lower']:.0f}, {row['ci_upper']:.0f}]")
    print()

    # --- Side-by-side comparison ---
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
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
          f"{'N/A':>12s}")


if __name__ == "__main__":
    main()
