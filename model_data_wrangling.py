import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_data(filepath):
    return pd.read_csv(filepath, delimiter=',')

def preprocess_date(df):
    df["date"] = pd.to_datetime(df["date"])
    df["date"].hist(bins=40)
    return df

def generate_advice_model(df):
    def advice_to_model(x):
        try:
            if x["home_team"] in x["advice"]:
                return x["advice"].replace(x["home_team"], "home")
            elif x["away_team"] in x["advice"]:
                return x["advice"].replace(x["away_team"], "away")
            else:
                return np.nan
        except:
            return np.nan
    df["advice_model"] = df.apply(advice_to_model, axis=1)
    df["advice_model"].hist(figsize=(50, 6))
    return df

def assign_winner(df):
    def winner(x):
        if x["match_home_goals"] > x["match_away_goals"]:
            return 0
        elif x["match_home_goals"] < x["match_away_goals"]:
            return 2
        else:
            return 1
    df["winner"] = df.apply(winner, axis=1)
    df[["winner"]].hist()
    return df

def classify_goals(df):
    df["total_goals"] = df['match_home_goals'] + df['match_away_goals']
    def goals_classes(x):
        if x < 3:
            return 0
        elif x < 5:
            return 1
        else:
            return 2
    df["goals_classes"] = df["total_goals"].apply(goals_classes)
    df["goals_classes"].hist()
    return df

def create_cumulative_stats(df, stats_columns):
    home_stats_columns = [x for x in stats_columns if "home" in x]
    away_stats_columns = [x for x in stats_columns if "away" in x]
    cum_stats_columns_home = ["cum_" + x + "_total" for x in home_stats_columns] + \
                             ["cum_" + x + "_avg" for x in home_stats_columns] + \
                             ["cum_" + x + "_by_minute" for x in home_stats_columns]
    cum_stats_columns_away = ["cum_" + x + "_total" for x in away_stats_columns] + \
                             ["cum_" + x + "_avg" for x in away_stats_columns] + \
                             ["cum_" + x + "_by_minute" for x in away_stats_columns]
    cum_stats_columns = cum_stats_columns_home + cum_stats_columns_away
    df_cum_stats = pd.DataFrame(columns=cum_stats_columns)

    for index, row in df.iterrows():
        season = row["season"]
        df_temp = df.loc[df["season"] == season].iloc[:index]

        def calculate_cum_stats(temp_df, stat_cols):
            matches = len(temp_df)
            minutes = matches * 90
            vals = temp_df[stat_cols].sum().values
            avg_vals = vals / matches if matches > 0 else np.zeros_like(vals)
            by_minute_vals = vals / minutes if minutes > 0 else np.zeros_like(vals)
            return np.concatenate((vals, avg_vals, by_minute_vals))

        temp_vals_home = calculate_cum_stats(df_temp.loc[df_temp["home_team"] == row["home_team"]], home_stats_columns)
        temp_vals_away = calculate_cum_stats(df_temp.loc[df_temp["away_team"] == row["away_team"]], away_stats_columns)
        temp_vals = np.concatenate((temp_vals_home, temp_vals_away))
        df_cum_stats_temp = pd.DataFrame([temp_vals], columns=cum_stats_columns)
        df_cum_stats = pd.concat([df_cum_stats, df_cum_stats_temp], axis=0)

    df_cum_stats = df_cum_stats.reset_index(drop=True)
    df_cum_stats = df_cum_stats.replace(np.nan, 0)
    return pd.concat([df, df_cum_stats], axis=1)

def create_division_columns(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns
    division_columns = []

    for feature in numeric_features:
        match_prueba = [x for x in numeric_features if x.replace("away", "home") == feature and x != feature]
        if match_prueba:
            division_columns.append([feature, match_prueba[0]])

    df_divisions = pd.DataFrame()

    for up_div, down_div in division_columns:
        new_column_name = f"{up_div}/{down_div}"
        df_divisions[new_column_name] = df[up_div] / df[down_div]

    return pd.concat([df, df_divisions], axis=1)

def main():
    df = load_data("model.csv")
    df = preprocess_date(df)
    df = generate_advice_model(df)
    df = assign_winner(df)
    df = classify_goals(df)
    stats_columns = [
        'shots_on_goal_home', 'shots_off_goal_home', 'total_shots_home', 'blocked_shots_home',
        'shots_insidebox_home', 'shots_outsidebox_home', 'fouls_home', 'corner_kicks_home',
        'offsides_home', 'ball_possession_home', 'yellow_cards_home', 'red_cards_home',
        'goalkeeper_saves_home', 'total_passes_home', 'passes_accurate_home', 'passes_%_home',
        'expected_goals_home', 'goals_prevented_home', 'shots_on_goal_away', 'shots_off_goal_away',
        'total_shots_away', 'blocked_shots_away', 'shots_insidebox_away', 'shots_outsidebox_away',
        'fouls_away', 'corner_kicks_away', 'offsides_away', 'ball_possession_away',
        'yellow_cards_away', 'red_cards_away', 'goalkeeper_saves_away', 'total_passes_away',
        'passes_accurate_away', 'passes_%_away', 'expected_goals_away', 'goals_prevented_away'
    ]
    df = create_cumulative_stats(df, stats_columns)
    df = create_division_columns(df)
    df.to_csv("df_for_model.csv", index=False)
    print("Data saved to df_for_model.csv")

def stats_columns():
    return  [
        'shots_on_goal_home', 'shots_off_goal_home', 'total_shots_home', 'blocked_shots_home',
        'shots_insidebox_home', 'shots_outsidebox_home', 'fouls_home', 'corner_kicks_home',
        'offsides_home', 'ball_possession_home', 'yellow_cards_home', 'red_cards_home',
        'goalkeeper_saves_home', 'total_passes_home', 'passes_accurate_home', 'passes_%_home',
        'expected_goals_home', 'goals_prevented_home', 'shots_on_goal_away', 'shots_off_goal_away',
        'total_shots_away', 'blocked_shots_away', 'shots_insidebox_away', 'shots_outsidebox_away',
        'fouls_away', 'corner_kicks_away', 'offsides_away', 'ball_possession_away',
        'yellow_cards_away', 'red_cards_away', 'goalkeeper_saves_away', 'total_passes_away',
        'passes_accurate_away', 'passes_%_away', 'expected_goals_away', 'goals_prevented_away'
    ]
if __name__ == "__main__":
    main()

load_data("outputs/master_model.csv")