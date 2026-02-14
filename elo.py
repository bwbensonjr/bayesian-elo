import math


class Elo:
    """
    Elo rating system for calculating team rankings and win probabilities.

    The Elo rating system assigns numerical ratings to teams and updates them based on
    game results. Higher-rated teams are expected to win more often, and ratings change
    based on whether outcomes match expectations.

    Attributes:
        ratings (dict): Dictionary mapping team names to their current Elo ratings.
        k (float): K-factor determining the maximum rating change per game.
        home_field (float): Home field advantage in Elo rating points.
        spread_factor (float): Divisor for converting Elo differences to point spreads.
        rating_mean (float): Historical mean rating for regression purposes.
    """

    def __init__(
            self,
            teams=None,
            k=3,
            home_field=20,
            spread_factor=55,
            rating_mean=1505,
    ):
        self.ratings = {}
        self.k = k
        self.home_field = home_field
        self.spread_factor = spread_factor
        self.rating_mean = rating_mean
        if teams:
            for team_name in teams:
                self.add_team(team_name)

    def add_team(self, team_name, initial_rating=1500):
        self.ratings[team_name] = initial_rating

    def team_rating(self, team_name):
        return self.ratings[team_name]

    def set_rating(self, team_name, new_rating):
        self.ratings[team_name] = new_rating

    def elo_difference(self, home_team, away_team):
        home_elo = self.team_rating(home_team)
        away_elo = self.team_rating(away_team)
        elo_diff = away_elo - (home_elo + self.home_field)
        return elo_diff

    def home_win_prob(self, home_team, away_team):
        elo_diff = self.elo_difference(home_team, away_team)
        expected_home = 1 / (1 + 10 ** (elo_diff / 400))
        return expected_home

    def point_spread(self, home_team, away_team):
        elo_diff = self.elo_difference(home_team, away_team)
        spread = -(elo_diff / self.spread_factor)
        return spread

    def update_ratings(self, home_team, home_score, away_team, away_score):
        if home_score > away_score:
            result_home = 1
        elif away_score > home_score:
            result_home = 0
        else:
            result_home = 0.5

        expected_home = self.home_win_prob(home_team, away_team)
        forecast_delta = result_home - expected_home

        score_diff = abs(home_score - away_score)
        score_multiplier = math.log(score_diff + 1)

        elo_change = self.k * score_multiplier * forecast_delta
        new_home_elo = self.team_rating(home_team) + elo_change
        new_away_elo = self.team_rating(away_team) - elo_change
        self.set_rating(home_team, new_home_elo)
        self.set_rating(away_team, new_away_elo)

        return new_home_elo, new_away_elo

    def regress_towards_mean(self, regress_mult=0.33):
        for team in self.ratings:
            old_rating = self.team_rating(team)
            rating_adjustment = (self.rating_mean - old_rating) * regress_mult
            new_rating = old_rating + rating_adjustment
            self.set_rating(team, new_rating)

    @staticmethod
    def calculate_mae(predicted, actual):
        errors = [abs(p - a) for p, a in zip(predicted, actual)]
        return sum(errors) / len(errors) if errors else 0.0

    @staticmethod
    def calculate_rmse(predicted, actual):
        squared_errors = [(p - a) ** 2 for p, a in zip(predicted, actual)]
        mse = sum(squared_errors) / len(squared_errors) if squared_errors else 0.0
        return math.sqrt(mse)

    def evaluate_predictions(self, games_df):
        valid_games = games_df[
            games_df["pred_point_spread"].notna() &
            games_df["actual_point_spread"].notna()
        ]

        if len(valid_games) == 0:
            return {"mae": 0.0, "rmse": 0.0, "count": 0}

        predicted = valid_games["pred_point_spread"].tolist()
        actual = valid_games["actual_point_spread"].tolist()

        return {
            "mae": self.calculate_mae(predicted, actual),
            "rmse": self.calculate_rmse(predicted, actual),
            "count": len(valid_games),
        }
