import pandas as pd

def main():
    df = read_premier_results()
    
def read_premier_results():
    """Read Premier League results and transform into
    desired format."""
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
            date = lambda x: x["date"].dt.date,
            actual_spread = lambda x: x["home_team_goals"] - x["away_team_goals"]
        )
    )
    return df
