import pandas as pd
from itertools import product

from elo import Elo


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


def main():
    df = read_premier_results()
    print(f"Loaded {len(df)} matches across {df['season'].nunique()} seasons")
    print()

    # Tune parameters
    print("=" * 60)
    print("PARAMETER TUNING")
    print("=" * 60)
    best_params, best_mae = tune_parameters(df)
    print()

    # Run with tuned parameters
    print("=" * 60)
    print("RESULTS WITH TUNED PARAMETERS")
    print("=" * 60)
    k = best_params["k"]
    hf = best_params["home_field"]
    sf = best_params["spread_factor"]
    print(f"Parameters: k={k}, home_field={hf}, spread_factor={sf}")
    print()

    results, elo = run_elo(df, k=k, home_field=hf, spread_factor=sf)

    # Overall metrics
    mae = Elo.calculate_mae(results["pred_spread"], results["actual_spread"])
    rmse = Elo.calculate_rmse(results["pred_spread"], results["actual_spread"])
    print(f"Overall MAE:  {mae:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print()

    # Win prediction accuracy
    results["pred_home_win"] = results["pred_home_win_prob"] > 0.5
    results["actual_home_win"] = results["actual_spread"] > 0
    # Exclude draws for win prediction accuracy
    non_draws = results[results["actual_spread"] != 0]
    accuracy = (non_draws["pred_home_win"] == non_draws["actual_home_win"]).mean()
    print(f"Win prediction accuracy (excl. draws): {accuracy:.1%}")
    print()

    # Per-season MAE
    print("Per-season MAE:")
    for season in sorted(results["season"].unique()):
        s = results[results["season"] == season]
        s_mae = Elo.calculate_mae(s["pred_spread"], s["actual_spread"])
        print(f"  {season}: {s_mae:.4f} ({len(s)} games)")
    print()

    # Final team rankings
    rankings = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)
    print("Final team rankings (top 10):")
    for i, (team, rating) in enumerate(rankings[:10], 1):
        print(f"  {i:2d}. {team:25s} {rating:.1f}")
    print()
    print("Final team rankings (bottom 10):")
    for i, (team, rating) in enumerate(rankings[-10:], len(rankings) - 9):
        print(f"  {i:2d}. {team:25s} {rating:.1f}")


if __name__ == "__main__":
    main()
