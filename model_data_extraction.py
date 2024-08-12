# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:51:57 2020

@author: USUARIO
"""
#%%
#Libraries
import requests
import json
import pandas as pd
import numpy as np
#import tqdm
from time import sleep
from datetime import datetime, date

#%%
#Headers
headers = {
    'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
    'x-rapidapi-key': "8d4e867913msh59f707e43f8b683p10a08fjsn9697ea1dc2c8"
    }
#%%
#Get last games

to_get = [[480, [2024]]
]
last_matches_columns = ["league", "fixture_id", "datetime",
                        "home_team", "away_team", "referee", "first_home_goals",
                        "first_away_goals", "match_home_goals", "match_away_goals"]

df_last_matches = pd.DataFrame(columns = last_matches_columns)
#%%
#Functions to need

def get_home_goals(x):
    if x == None:
        return 0
    else:
        x = int(x[0:x.find("-")])
        return x
def get_away_goals(x):
    if x == None:
        return 0
    else:
        x = int(x[x.find("-")+1:])
        return x
def get_round(x):
    try:
        x=int(x[x.find("-")+2:])
        return x
    except:
        return x
#%%
failed_last_matches  = []
for league in to_get:
    for rnd in league[1]:
        print(league[0], rnd)
        url_round = "https://api-football-v1.p.rapidapi.com/v3/fixtures?league=" + str(league[0]) + "&season=" + str(rnd)
        #print(url_round)
        response_round = requests.request("GET", url_round, headers=headers)
        sleep(1.5)
        round_fixtures = json.loads(response_round.text)
        fixtures = round_fixtures["response"]
        for fixture in fixtures:
    
 
            df_temp = pd.DataFrame(data = [[fixture['league']['name'],
                                            fixture["fixture"]["id"],
                                            fixture["fixture"]["date"],
                                            fixture["teams"]["home"]["name"],
                                            fixture["teams"]["away"]["name"],
                                            fixture["fixture"]['referee'],
                                            fixture['score']['halftime']['home'],
                                            fixture['score']['halftime']['away'],
                                            fixture['goals']['home'],
                                            fixture['goals']['away']]], columns =last_matches_columns)
            
            df_last_matches = df_last_matches._append(df_temp)
#%%
df_last_matches["match_away_goals"] = df_last_matches["match_away_goals"].apply(lambda x: 0 if x==None else x)
df_last_matches["match_home_goals"] = df_last_matches["match_home_goals"].apply(lambda x: 0 if x==None else x)
df_last_matches = df_last_matches.reset_index(drop=True)
df_last_matches

print(df_last_matches.head(), df_last_matches.columns, df_last_matches.shape)
print(failed_last_matches)
df_last_matches.to_csv("temp_historical_2012.csv", index = False)

#%%
df_last_matches = pd.read_csv("temp_historical_2012.csv")
df_last_matches["datetime"] = df_last_matches["datetime"].apply(lambda x: x.replace("T", " "))
df_last_matches["datetime"] = pd.to_datetime(df_last_matches["datetime"], utc=True)
df_last_matches["time"] = df_last_matches["datetime"].dt.time 
df_last_matches["date"] = df_last_matches["datetime"].dt.date
df_last_matches = df_last_matches.sort_values("datetime")
df_last_matches = df_last_matches.reset_index(drop = True)
df_last_matches = df_last_matches.loc[df_last_matches["date"]<=date.today()]
#%%
def get_season(x):
    if x.month < 7:
        return x.year-1
    else:
        return x.year
    
df_last_matches["season"] = df_last_matches["date"].apply(get_season)
#df_last_matches["season"] = df_last_matches["season"].apply(lambda x: 2012 if x<2012 else x)
df_last_matches["season"]
#%%
#df_last_matches.loc[df_last_matches.season==2011][["date", "league", "home_team", "away_team", "round"]]

#%%
for i in df_last_matches.league.unique():
    print(i)
#%%
"""
leagues = ["Championship", "Bundesliga 2", "Segunda Division", "Serie B"]
df_last_matches = df_last_matches.loc[df_last_matches["league"].isin(leagues)]

seasons = [2017, 2018, 2019]
df_last_matches = df_last_matches.loc[df_last_matches["season"].isin(seasons)]

"""
#%%
#df_last_matches.loc[df_last_matches["home_team"] == "Fuenlabrada"][["fixture_id", "away_team"]]

#%%
failed_predictions = []
#Get predictions per game
prediction_cols = ["fixture_id", 'match_winner', 'win_or_draw', 'under_over', "advice", "goals_home",
                   "goals_away", "home_pred", "draw_pred", "away_pred",
                   
                   "home_last5_forme", "home_last5_att", "home_last5_def",
                   "home_last5_goals", "home_last5_goals_avg", "home_last5_goals_against",
                   "home_last5_goals_against_avg", "home_matches_played_home",
                   "home_matches_played_away", "home_matches_played_total", "home_matches_won_home",
                   "home_matches_won_away", "home_matches_won_total", "home_matches_draw_home",
                   "home_matches_draw_away", "home_matches_draw_total","home_matches_lost_home",
                   "home_matches_lost_away", "home_matches_lost_total", 
                   "home_goals_fav_home", "home_goals_fav_away", "home_goals_fav_total",
                   "home_goals_ag_home", "home_goals_ag_away", "home_goals_ag_total",
                   "home_goalsavg_fav_home", "home_goalsavg_fav_away", "home_goalsavg_fav_total",
                   "home_goalsavg_ag_home", "home_goalsavg_ag_away", "home_goalsavg_ag_total",
                   
                   "away_last5_forme", "away_last5_att", "away_last5_def",
                   "away_last5_goals", "away_last5_goals_avg", "away_last5_goals_against",
                   "away_last5_goals_against_avg", "away_matches_played_home",
                   "away_matches_played_away", "away_matches_played_total", "away_matches_won_home",
                   "away_matches_won_away", "away_matches_won_total", "away_matches_draw_home",
                   "away_matches_draw_away", "away_matches_draw_total","away_matches_lost_home",
                   "away_matches_lost_away", "away_matches_lost_total", 
                   "away_goals_fav_home", "away_goals_fav_away", "away_goals_fav_total",
                   "away_goals_ag_home", "away_goals_ag_away", "away_goals_ag_total",
                   "away_goalsavg_fav_home", "away_goalsavg_fav_away", "away_goalsavg_fav_total",
                   "away_goalsavg_ag_home", "away_goalsavg_ag_away", "away_goalsavg_ag_total",
                   
                   "last_h2h_home_home_total", "last_h2h_away_home_total", "last_h2h_total",
                   "last_h2h_home_won_home", "last_h2h_home_won_away", "last_h2h_home_won_total",
                   "last_h2h_away_won_home", "last_h2h_away_won_away", "last_h2h_away_won_total",
                   "last_h2h_draw_home_home", "last_h2h_draw_away_home", "last_h2h_draw_away_total",
                   
                   "comparison_forme_home", "comparison_forme_away",
                   "comparison_att_home", "comparison_att_away",
                   "comparison_def_home", "comparison_def_away",
                   "comparison_fishlaw_home", "comparison_fishlaw_away",
                   "comparison_h2h_home", "comparison_h2h_away",
                   "comparison_goalsh2h_home", "comparison_goalsh2h_away"
                   ]
#%%
df_predictions = pd.DataFrame(columns = prediction_cols) 

for index, row in df_last_matches.iloc[:2].iterrows():

    print(i)
    fixture_id = row["fixture_id"]
    league = row["league"]
    print(fixture_id, league)
    
    
    url_predictions = "https://api-football-v1.p.rapidapi.com/v3/predictions?fixture=" + str(fixture_id)
    response_predictions = requests.request("GET", url_predictions, headers=headers)
    sleep(1.5)
    prediction = json.loads(response_predictions.text)


    # Accessing the predictions
    match_winner = prediction["response"][0]["predictions"]["winner"]["name"]
    win_or_draw = prediction["response"][0]["predictions"]["win_or_draw"]
    under_over = prediction["response"][0]["predictions"]["under_over"]
    advice = prediction["response"][0]["predictions"]["advice"]

    try:
        goals_home = float(prediction["response"][0]["predictions"]["goals"]["home"])
        goals_away = float(prediction["response"][0]["predictions"]["goals"]["away"])
    except:
        goals_home = np.nan
        goals_away = np.nan

    home_pred = float(prediction["response"][0]["predictions"]["percent"]["home"].strip('%')) / 100
    draw_pred = float(prediction["response"][0]["predictions"]["percent"]["draw"].strip('%')) / 100
    away_pred = float(prediction["response"][0]["predictions"]["percent"]["away"].strip('%')) / 100

    # Accessing the last 5 matches for the home team
    home_last5_forme = float(prediction["response"][0]["teams"]["home"]["last_5"]["form"].strip('%')) / 100
    home_last5_att = float(prediction["response"][0]["teams"]["home"]["last_5"]["att"].strip('%')) / 100
    home_last5_def = float(prediction["response"][0]["teams"]["home"]["last_5"]["def"].strip('%')) / 100
    home_last5_goals = float(prediction["response"][0]["teams"]["home"]["last_5"]["goals"]["for"]["total"])
    home_last5_goals_avg = float(prediction["response"][0]["teams"]["home"]["last_5"]["goals"]["for"]["average"])
    home_last5_goals_against = float(prediction["response"][0]["teams"]["home"]["last_5"]["goals"]["against"]["total"])
    home_last5_goals_against_avg = float(prediction["response"][0]["teams"]["home"]["last_5"]["goals"]["against"]["average"])

    # Accessing the last 5 matches for the away team
    away_last5_forme = float(prediction["response"][0]["teams"]["away"]["last_5"]["form"].strip('%')) / 100
    away_last5_att = float(prediction["response"][0]["teams"]["away"]["last_5"]["att"].strip('%')) / 100
    away_last5_def = float(prediction["response"][0]["teams"]["away"]["last_5"]["def"].strip('%')) / 100
    away_last5_goals = float(prediction["response"][0]["teams"]["away"]["last_5"]["goals"]["for"]["total"])
    away_last5_goals_avg = float(prediction["response"][0]["teams"]["away"]["last_5"]["goals"]["for"]["average"])
    away_last5_goals_against = float(prediction["response"][0]["teams"]["away"]["last_5"]["goals"]["against"]["total"])
    away_last5_goals_against_avg = float(prediction["response"][0]["teams"]["away"]["last_5"]["goals"]["against"]["average"])

    # Accessing the all last matches for the home team
    home_matches_played_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["played"]["home"])
    home_matches_played_away = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["played"]["away"])
    home_matches_played_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["played"]["total"])
    home_matches_won_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["home"])
    home_matches_won_away = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["away"])
    home_matches_won_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["total"])
    home_matches_draw_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["home"])
    home_matches_draw_away = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["away"])
    home_matches_draw_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["total"])
    home_matches_lost_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["loses"]["home"])
    home_matches_lost_away = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["loses"]["away"])
    home_matches_lost_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["loses"]["total"])
    home_goals_fav_home = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["total"]["home"])
    home_goals_fav_away = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["total"]["away"])
    home_goals_fav_total = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["total"]["total"])
    home_goals_ag_home = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["total"]["home"])
    home_goals_ag_away = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["total"]["away"])
    home_goals_ag_total = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["total"]["total"])
    home_goalsavg_fav_home = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["average"]["home"])
    home_goalsavg_fav_away = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["average"]["away"])
    home_goalsavg_fav_total = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["for"]["average"]["total"])
    home_goalsavg_ag_home = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["average"]["home"])
    home_goalsavg_ag_away = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["average"]["away"])
    home_goalsavg_ag_total = float(prediction["response"][0]["teams"]["home"]["league"]["goals"]["against"]["average"]["total"])

    # Accessing the all last matches for the away team
    away_matches_played_home = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["played"]["home"])
    away_matches_played_away = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["played"]["away"])
    away_matches_played_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["played"]["total"])
    away_matches_won_home = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["home"])
    away_matches_won_away = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["away"])
    away_matches_won_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["total"])
    away_matches_draw_home = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["draws"]["home"])
    away_matches_draw_away = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["draws"]["away"])
    away_matches_draw_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["draws"]["total"])
    away_matches_lost_home = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["loses"]["home"])
    away_matches_lost_away = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["loses"]["away"])
    away_matches_lost_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["loses"]["total"])
    away_goals_fav_home = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["total"]["home"])
    away_goals_fav_away = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["total"]["away"])
    away_goals_fav_total = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["total"]["total"])
    away_goals_ag_home = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["total"]["home"])
    away_goals_ag_away = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["total"]["away"])
    away_goals_ag_total = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["total"]["total"])
    away_goalsavg_fav_home = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["average"]["home"])
    away_goalsavg_fav_away = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["average"]["away"])
    away_goalsavg_fav_total = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["for"]["average"]["total"])
    away_goalsavg_ag_home = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["average"]["home"])
    away_goalsavg_ag_away = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["average"]["away"])
    away_goalsavg_ag_total = float(prediction["response"][0]["teams"]["away"]["league"]["goals"]["against"]["average"]["total"])

    # Accessing the last head-to-head matches
    last_h2h_home_home_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["played"]["home"])
    last_h2h_away_home_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["played"]["home"])
    last_h2h_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["played"]["total"])
    last_h2h_home_won_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["home"])
    last_h2h_home_won_away = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["away"])
    last_h2h_home_won_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["wins"]["total"])
    last_h2h_away_won_home = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["home"])
    last_h2h_away_won_away = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["away"])
    last_h2h_away_won_total = float(prediction["response"][0]["teams"]["away"]["league"]["fixtures"]["wins"]["total"])
    last_h2h_draw_home_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["home"])
    last_h2h_draw_away_home = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["away"])
    last_h2h_draw_away_total = float(prediction["response"][0]["teams"]["home"]["league"]["fixtures"]["draws"]["total"])

    # Accessing the comparison metrics
    comparison_forme_home = float(prediction["response"][0]["comparison"]["form"]["home"].strip('%')) / 100
    comparison_forme_away = float(prediction["response"][0]["comparison"]["form"]["away"].strip('%')) / 100
    comparison_att_home = float(prediction["response"][0]["comparison"]["att"]["home"].strip('%')) / 100
    comparison_att_away = float(prediction["response"][0]["comparison"]["att"]["away"].strip('%')) / 100
    comparison_def_home = float(prediction["response"][0]["comparison"]["def"]["home"].strip('%')) / 100
    comparison_def_away = float(prediction["response"][0]["comparison"]["def"]["away"].strip('%')) / 100
    comparison_fishlaw_home = float(prediction["response"][0]["comparison"]["poisson_distribution"]["home"].strip('%')) / 100
    comparison_fishlaw_away = float(prediction["response"][0]["comparison"]["poisson_distribution"]["away"].strip('%')) / 100
    comparison_h2h_home = float(prediction["response"][0]["comparison"]["h2h"]["home"].strip('%')) / 100
    comparison_h2h_away = float(prediction["response"][0]["comparison"]["h2h"]["away"].strip('%')) / 100
    comparison_goalsh2h_home = float(prediction["response"][0]["comparison"]["goals"]["home"].strip('%')) / 100
    comparison_goalsh2h_away = float(prediction["response"][0]["comparison"]["goals"]["away"].strip('%')) / 100
    #%%
    data = [[fixture_id, match_winner, win_or_draw, under_over, advice, goals_home,
                goals_away, home_pred, draw_pred, away_pred,
                
                home_last5_forme, home_last5_att, home_last5_def,
                home_last5_goals, home_last5_goals_avg, home_last5_goals_against,
                home_last5_goals_against_avg, home_matches_played_home,
                home_matches_played_away, home_matches_played_total, home_matches_won_home,
                home_matches_won_away, home_matches_won_total, home_matches_draw_home,
                home_matches_draw_away, home_matches_draw_total,home_matches_lost_home,
                home_matches_lost_away, home_matches_lost_total,
                home_goals_fav_home, home_goals_fav_away, home_goals_fav_total,
                home_goals_ag_home, home_goals_ag_away, home_goals_ag_total,
                home_goalsavg_fav_home, home_goalsavg_fav_away, home_goalsavg_fav_total,
                home_goalsavg_ag_home, home_goalsavg_ag_away, home_goalsavg_ag_total,
                
                away_last5_forme, away_last5_att, away_last5_def,
                away_last5_goals, away_last5_goals_avg, away_last5_goals_against,
                away_last5_goals_against_avg, away_matches_played_home,
                away_matches_played_away, away_matches_played_total, away_matches_won_home,
                away_matches_won_away, away_matches_won_total, away_matches_draw_home,
                away_matches_draw_away, away_matches_draw_total,away_matches_lost_home,
                away_matches_lost_away, away_matches_lost_total,
                away_goals_fav_home, away_goals_fav_away, away_goals_fav_total,
                away_goals_ag_home, away_goals_ag_away, away_goals_ag_total,
                away_goalsavg_fav_home, away_goalsavg_fav_away, away_goalsavg_fav_total,
                away_goalsavg_ag_home, away_goalsavg_ag_away, away_goalsavg_ag_total,
                
                last_h2h_home_home_total, last_h2h_away_home_total, last_h2h_total,
                last_h2h_home_won_home, last_h2h_home_won_away, last_h2h_home_won_total,
                last_h2h_away_won_home, last_h2h_away_won_away, last_h2h_away_won_total,
                last_h2h_draw_home_home, last_h2h_draw_away_home, last_h2h_draw_away_total,
                
                comparison_forme_home, comparison_forme_away,
                comparison_att_home, comparison_att_away,
                comparison_def_home, comparison_def_away,
                comparison_fishlaw_home, comparison_fishlaw_away,
                comparison_h2h_home, comparison_h2h_away,
                comparison_goalsh2h_home, comparison_goalsh2h_away
                ]]

    #%%
    df_temp = pd.DataFrame(data,
                            columns = prediction_cols) 
    df_predictions = df_predictions._append(df_temp)
 

df_predictions = df_predictions.reset_index(drop=True)
df_predictions
df_predictions.to_csv("temp_historical_2012_predictions.csv", index = False)


#%%
df_stats = pd.DataFrame()

failed_stats = []
for index, row in df_last_matches[:2].iterrows():
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
print(df_stats)
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

drop_columns = [col for col in df_stats.columns if col not in stats_columns]
drop_columns
#%%
# Fill missing values and convert percentages to floats
df_stats = df_stats.fillna(0).replace("None", 0).reset_index(drop=True)
for col in df_stats.columns:
    try:
        df_stats[col] = df_stats[col].astype(float)
    except ValueError:
        df_stats[col] = df_stats[col].apply(lambda x: float(str(x).strip('%')) / 100)

# Ensure fixture_id is an integer
df_stats["fixture_id"] = df_stats["fixture_id"].astype(int)

# Save to CSV
df_stats.to_csv("temp_historical_2012_stats.csv", index=False)
#%%

df_final = df_last_matches.merge(df_predictions, on = "fixture_id", how = "left")
df_final = df_final.merge(df_stats, on = "fixture_id", how = "left")


"""
df_old = pd.read_csv("master_model.csv")
print(len(df_old))

df_final = pd.concat([df_final, df_old], axis = 0)
print(df_final.head(), df_final.columns, df_final.shape)
"""
#%%
df_final = df_final.drop_duplicates("fixture_id", keep = "last")
df_final = df_final.fillna(0)
df_final.to_csv("master_model.csv", index=False)


# %%
