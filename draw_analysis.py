"""
Analysis of draw prevalence in Premier League matches and whether
higher-rated matchups produce more draws.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from elo import Elo
from premier_league import read_premier_results


def build_elo_ratings_per_match(df):
    """Run Elo sequentially, recording both teams' ratings BEFORE each match."""
    df = df.sort_values(["season", "date"]).reset_index(drop=True)
    elo = Elo(k=3, home_field=20, spread_factor=55)

    home_ratings = []
    away_ratings = []
    seasons = df["season"].unique()

    for season in seasons:
        season_mask = df["season"] == season
        season_games = df[season_mask]

        for idx, row in season_games.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            if home not in elo.ratings:
                elo.add_team(home)
            if away not in elo.ratings:
                elo.add_team(away)

            home_ratings.append(elo.team_rating(home))
            away_ratings.append(elo.team_rating(away))

            elo.update_ratings(
                home, row["home_team_goals"],
                away, row["away_team_goals"],
            )

        elo.regress_towards_mean()

    df = df.copy()
    df["home_elo"] = home_ratings
    df["away_elo"] = away_ratings
    df["avg_elo"] = (df["home_elo"] + df["away_elo"]) / 2
    df["elo_diff"] = (df["home_elo"] - df["away_elo"]).abs()
    df["is_draw"] = (df["result"] == "D").astype(int)
    return df


def overall_draw_stats(df):
    """Print overall draw prevalence."""
    total = len(df)
    draws = df["is_draw"].sum()
    print("=" * 60)
    print("OVERALL DRAW PREVALENCE")
    print("=" * 60)
    print(f"Total matches: {total}")
    print(f"Draws:         {draws} ({100 * draws / total:.1f}%)")
    print(f"Home wins:     {(df['result'] == 'H').sum()} ({100 * (df['result'] == 'H').mean():.1f}%)")
    print(f"Away wins:     {(df['result'] == 'A').sum()} ({100 * (df['result'] == 'A').mean():.1f}%)")
    print()


def draw_rate_by_season(df):
    """Print and plot draw rate by season."""
    season_stats = (
        df.groupby("season")
        .agg(
            matches=("is_draw", "count"),
            draws=("is_draw", "sum"),
        )
        .assign(draw_rate=lambda x: x["draws"] / x["matches"])
    )

    print("=" * 60)
    print("DRAW RATE BY SEASON")
    print("=" * 60)
    print(season_stats.to_string())
    print()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(season_stats)), season_stats["draw_rate"], color="steelblue", alpha=0.8)
    ax.axhline(df["is_draw"].mean(), color="red", linestyle="--", label=f"Overall: {df['is_draw'].mean():.1%}")
    ax.set_xticks(range(len(season_stats)))
    ax.set_xticklabels(season_stats.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Draw Rate")
    ax.set_title("Premier League Draw Rate by Season (1993-2022)")
    ax.legend()
    ax.set_ylim(0, 0.40)
    fig.tight_layout()
    fig.savefig("data/draw_rate_by_season.png", dpi=150)
    print("Saved: data/draw_rate_by_season.png\n")


def draw_rate_by_avg_elo(df):
    """Analyze draw rate by average Elo of the two teams (match quality)."""
    # Skip first season where all teams are at 1500
    df_rated = df[df["season"] != df["season"].unique()[0]].copy()

    # Bin by average Elo quintiles
    df_rated["avg_elo_bin"] = pd.qcut(df_rated["avg_elo"], 5, labels=False)
    bin_edges = pd.qcut(df_rated["avg_elo"], 5, retbins=True)[1]

    bin_stats = (
        df_rated.groupby("avg_elo_bin")
        .agg(
            matches=("is_draw", "count"),
            draws=("is_draw", "sum"),
            avg_elo_mean=("avg_elo", "mean"),
        )
        .assign(draw_rate=lambda x: x["draws"] / x["matches"])
    )

    print("=" * 60)
    print("DRAW RATE BY AVERAGE ELO (quintiles)")
    print("Higher avg Elo = both teams are stronger")
    print("=" * 60)
    for i, row in bin_stats.iterrows():
        lo, hi = bin_edges[i], bin_edges[i + 1]
        print(f"  Q{i+1} (Elo {lo:.0f}-{hi:.0f}): "
              f"{row['draw_rate']:.1%} draws ({int(row['draws'])}/{int(row['matches'])} matches)")
    print()

    # Correlation test
    r, p = stats.pointbiserialr(df_rated["is_draw"], df_rated["avg_elo"])
    print(f"Point-biserial correlation (draw vs avg_elo): r={r:.4f}, p={p:.4f}")
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"Q{i+1}\n({bin_edges[i]:.0f}-{bin_edges[i+1]:.0f})" for i in range(5)]
    ax.bar(labels, bin_stats["draw_rate"], color="steelblue", alpha=0.8)
    ax.axhline(df_rated["is_draw"].mean(), color="red", linestyle="--",
               label=f"Overall: {df_rated['is_draw'].mean():.1%}")
    ax.set_ylabel("Draw Rate")
    ax.set_xlabel("Average Elo of Both Teams (quintile)")
    ax.set_title("Draw Rate vs Match Quality (Average Elo)")
    ax.legend()
    ax.set_ylim(0, 0.40)
    fig.tight_layout()
    fig.savefig("data/draw_rate_by_avg_elo.png", dpi=150)
    print("Saved: data/draw_rate_by_avg_elo.png\n")


def draw_rate_by_elo_diff(df):
    """Analyze draw rate by Elo difference (match competitiveness)."""
    df_rated = df[df["season"] != df["season"].unique()[0]].copy()

    # Bin by absolute Elo difference quintiles
    df_rated["elo_diff_bin"] = pd.qcut(df_rated["elo_diff"], 5, labels=False, duplicates="drop")
    bin_edges = pd.qcut(df_rated["elo_diff"], 5, retbins=True, duplicates="drop")[1]

    bin_stats = (
        df_rated.groupby("elo_diff_bin")
        .agg(
            matches=("is_draw", "count"),
            draws=("is_draw", "sum"),
            elo_diff_mean=("elo_diff", "mean"),
        )
        .assign(draw_rate=lambda x: x["draws"] / x["matches"])
    )

    print("=" * 60)
    print("DRAW RATE BY ELO DIFFERENCE (quintiles)")
    print("Smaller diff = more evenly matched teams")
    print("=" * 60)
    for i, row in bin_stats.iterrows():
        lo, hi = bin_edges[i], bin_edges[i + 1]
        print(f"  Q{i+1} (diff {lo:.0f}-{hi:.0f}): "
              f"{row['draw_rate']:.1%} draws ({int(row['draws'])}/{int(row['matches'])} matches)")
    print()

    r, p = stats.pointbiserialr(df_rated["is_draw"], df_rated["elo_diff"])
    print(f"Point-biserial correlation (draw vs elo_diff): r={r:.4f}, p={p:.4f}")
    print()

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"Q{i+1}\n({bin_edges[i]:.0f}-{bin_edges[i+1]:.0f})" for i in range(len(bin_stats))]
    ax.bar(labels, bin_stats["draw_rate"], color="darkorange", alpha=0.8)
    ax.axhline(df_rated["is_draw"].mean(), color="red", linestyle="--",
               label=f"Overall: {df_rated['is_draw'].mean():.1%}")
    ax.set_ylabel("Draw Rate")
    ax.set_xlabel("Absolute Elo Difference (quintile)")
    ax.set_title("Draw Rate vs Match Competitiveness (Elo Difference)")
    ax.legend()
    ax.set_ylim(0, 0.40)
    fig.tight_layout()
    fig.savefig("data/draw_rate_by_elo_diff.png", dpi=150)
    print("Saved: data/draw_rate_by_elo_diff.png\n")


def draw_rate_combined(df):
    """2D analysis: avg Elo (quality) x Elo diff (competitiveness)."""
    df_rated = df[df["season"] != df["season"].unique()[0]].copy()

    df_rated["quality"] = pd.qcut(df_rated["avg_elo"], 3, labels=["Low", "Mid", "High"])
    df_rated["closeness"] = pd.qcut(df_rated["elo_diff"], 3, labels=["Close", "Mid", "Mismatch"])

    pivot = df_rated.pivot_table(
        values="is_draw", index="quality", columns="closeness", aggfunc="mean"
    )

    print("=" * 60)
    print("DRAW RATE: MATCH QUALITY x COMPETITIVENESS")
    print("(rows = avg Elo tercile, cols = Elo diff tercile)")
    print("=" * 60)
    print(pivot.map(lambda x: f"{x:.1%}").to_string())
    print()

    counts = df_rated.pivot_table(
        values="is_draw", index="quality", columns="closeness", aggfunc="count"
    )
    print("Match counts:")
    print(counts.to_string())
    print()


def logistic_regression_analysis(df):
    """Logistic regression: P(draw) ~ avg_elo + elo_diff using scipy."""
    from scipy.optimize import minimize

    df_rated = df[df["season"] != df["season"].unique()[0]].copy()

    # Standardize features
    avg_elo = df_rated["avg_elo"].values
    elo_diff = df_rated["elo_diff"].values
    y = df_rated["is_draw"].values.astype(float)

    avg_elo_z = (avg_elo - avg_elo.mean()) / avg_elo.std()
    elo_diff_z = (elo_diff - elo_diff.mean()) / elo_diff.std()
    X = np.column_stack([np.ones(len(y)), avg_elo_z, elo_diff_z])

    def neg_log_likelihood(beta):
        z = X @ beta
        z = np.clip(z, -500, 500)
        p = 1 / (1 + np.exp(-z))
        ll = y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
        return -ll.sum()

    result = minimize(neg_log_likelihood, np.zeros(3), method="L-BFGS-B")
    intercept, b_avg_elo, b_elo_diff = result.x

    print("=" * 60)
    print("LOGISTIC REGRESSION: P(draw) ~ avg_elo + elo_diff")
    print("(features standardized)")
    print("=" * 60)
    print(f"  avg_elo coefficient:  {b_avg_elo:+.4f}")
    print(f"  elo_diff coefficient: {b_elo_diff:+.4f}")
    print(f"  intercept:            {intercept:+.4f}")
    print()
    print("Interpretation:")
    print("  Positive avg_elo coeff => higher-rated matchups draw more")
    print("  Negative elo_diff coeff => closer matches draw more")
    print()


def top_team_draw_analysis(df):
    """Compare draw rates when top-6 teams play each other vs other matches."""
    # Identify top teams by final Elo rating
    top_teams_by_era = {
        "traditional_top6": ["Man United", "Arsenal", "Chelsea", "Liverpool", "Man City", "Tottenham"],
    }

    df_rated = df[df["season"] != df["season"].unique()[0]].copy()
    top6 = top_teams_by_era["traditional_top6"]

    df_rated["home_top6"] = df_rated["home_team"].isin(top6)
    df_rated["away_top6"] = df_rated["away_team"].isin(top6)
    df_rated["match_type"] = "Other"
    df_rated.loc[df_rated["home_top6"] & df_rated["away_top6"], "match_type"] = "Top6 vs Top6"
    df_rated.loc[
        (df_rated["home_top6"] | df_rated["away_top6"]) &
        ~(df_rated["home_top6"] & df_rated["away_top6"]),
        "match_type"
    ] = "Top6 vs Non-Top6"

    print("=" * 60)
    print("DRAW RATE BY MATCH TYPE (Traditional Top 6)")
    print(f"Top 6: {', '.join(top6)}")
    print("=" * 60)
    for mt in ["Top6 vs Top6", "Top6 vs Non-Top6", "Other"]:
        subset = df_rated[df_rated["match_type"] == mt]
        dr = subset["is_draw"].mean()
        print(f"  {mt:20s}: {dr:.1%} draws ({subset['is_draw'].sum()}/{len(subset)} matches)")
    print()

    # Chi-squared test
    contingency = pd.crosstab(df_rated["match_type"], df_rated["is_draw"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"Chi-squared test: chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
    print()


def main():
    print("Loading Premier League data...\n")
    df = read_premier_results()

    print("Building Elo ratings for each match...\n")
    df = build_elo_ratings_per_match(df)

    overall_draw_stats(df)
    draw_rate_by_season(df)
    draw_rate_by_avg_elo(df)
    draw_rate_by_elo_diff(df)
    draw_rate_combined(df)
    top_team_draw_analysis(df)
    logistic_regression_analysis(df)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
This analysis examined whether draws are more common when
higher-rated teams face each other in the Premier League.

Two dimensions of "team quality" were tested:
  1. Average Elo (match quality): Are draws more likely when
     both teams are strong?
  2. Elo difference (competitiveness): Are draws more likely
     when teams are evenly matched?

See the correlation statistics and logistic regression above
for the quantitative results.
""")


if __name__ == "__main__":
    main()
