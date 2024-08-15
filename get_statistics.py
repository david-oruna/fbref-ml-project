import requests
import json
import pandas as pd
import numpy as np
import os
from time import sleep
from datetime import datetime, date

#%%
headers = {
    'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
    'x-rapidapi-key': os.getenv("RAPIDAPI_KEY")
    }

df_stats = pd.DataFrame()
df_last_matches = pd.read_csv("fixtures.csv")
failed_stats = []
for index, row in df_last_matches[:1].iterrows():
    fixture_id = row["fixture_id"]
    league = row["league"]
    print(fixture_id, league)
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics?fixture=" + str(fixture_id)
    response_stats = requests.request("GET", url, headers=headers)
    sleep(1.5)
    stats = json.loads(response_stats.text)



    home_team_stats = stats['response'][0]['statistics']
    away_team_stats = stats['response'][1]['statistics']

    # Convert statistics lists to dictionaries
    home_stats_dict = {stat['type'].replace(" ", "_").lower(): stat['value'] for stat in home_team_stats}
    away_stats_dict = {stat['type'].replace(" ", "_").lower(): stat['value'] for stat in away_team_stats}

    # Add prefixes to distinguish home and away stats
    home_stats_dict = {f"{key}_home": value for key, value in home_stats_dict.items()}
    away_stats_dict = {f"{key}_away": value for key, value in away_stats_dict.items()}

    # Combine home and away stats into a single dictionary
    combined_stats_dict = {**home_stats_dict, **away_stats_dict}
    
    # Add fixture_id to the combined stats dictionary
    combined_stats_dict['fixture_id'] = fixture_id

    # Convert dictionary to DataFrame and concatenate
    df_temp = pd.DataFrame([combined_stats_dict])
    df_stats = pd.concat([df_stats, df_temp], ignore_index=True)

#%%
stats_columns = [
    'shots_on_goal_home', 'shots_off_goal_home', 'total_shots_home', 'blocked_shots_home',
    'shots_insidebox_home', 'shots_outsidebox_home', 'fouls_home', 'corner_kicks_home',
    'offsides_home', 'ball_possession_home', 'yellow_cards_home', 'red_cards_home',
    'goalkeeper_saves_home', 'total_passes_home', 'passes_accurate_home', 'passes_%_home',
    'expected_goals_home', 'goals_prevented_home',

    'shots_on_goal_away', 'shots_off_goal_away', 'total_shots_away', 'blocked_shots_away',
    'shots_insidebox_away', 'shots_outsidebox_away', 'fouls_away', 'corner_kicks_away',
    'offsides_away', 'ball_possession_away', 'yellow_cards_away', 'red_cards_away',
    'goalkeeper_saves_away', 'total_passes_away', 'passes_accurate_away', 'passes_%_away',
    'expected_goals_away', 'goals_prevented_away','fixture_id'
]


#%%
# Fill missing values and convert percentages to floats
df_stats = df_stats.fillna(0).replace("None", 0).reset_index(drop=True)
for col in df_stats.columns:
    try:
        df_stats[col] = df_stats[col].astype(float)
    except ValueError:
        df_stats[col] = df_stats[col].apply(lambda x: float(str(x).strip('%')) / 100)

# Save to CSV
df_stats.to_csv("stats.csv", index=False)
#%%

