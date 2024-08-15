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
from time import sleep
from datetime import datetime, date
import os

#%%
#Headers
headers = {
    'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
    'x-rapidapi-key': os.getenv("RAPIDAPI_KEY")
    }

#%%
#Get last games
def get_last_matches(to_get):
    last_matches_columns = ["league", "fixture_id", "datetime",
                            "home_team", "away_team", "referee", "first_home_goals",
                            "first_away_goals", "match_home_goals", "match_away_goals"]

    df_last_matches = pd.DataFrame(columns = last_matches_columns)

    for league in to_get:
        for rnd in league[1]:
            print(league[0], rnd)
            url_round = "https://api-football-v1.p.rapidapi.com/v3/fixtures?league=" + str(league[0]) + "&season=" + str(rnd)
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
                df_last_matches = df_last_matches.append(df_temp)

    df_last_matches["match_away_goals"] = df_last_matches["match_away_goals"].apply(lambda x: 0 if x==None else x)
    df_last_matches["match_home_goals"] = df_last_matches["match_home_goals"].apply(lambda x: 0 if x==None else x)
    df_last_matches["datetime"] = df_last_matches["datetime"].apply(lambda x: x.replace("T", " "))
    df_last_matches["datetime"] = pd.to_datetime(df_last_matches["datetime"], utc=True)
    df_last_matches["time"] = df_last_matches["datetime"].dt.time 
    df_last_matches["date"] = df_last_matches["datetime"].dt.date
    df_last_matches["date"] = df_last_matches["date"].dt.date

    df_last_matches = df_last_matches.sort_values("datetime")
    df_last_matches = df_last_matches.reset_index(drop = True)
    df_last_matches = df_last_matches.loc[df_last_matches["date"]<date.today()]
    def get_season(x):
        if x.month < 7:
            return x.year-1
        else:
            return x.year
    
    df_last_matches["season"] = df_last_matches["date"].apply(get_season)
    df_last_matches["fixture_id"] = df_last_matches["fixture_id"].astype(int)
    return df_last_matches

#%%
#Save last matches to CSV
def save_last_matches(df_last_matches, path = "master_model.csv"):
    try:
        df_old = pd.read_csv(path)
        df_final = pd.concat([df_last_matches, df_old], axis = 0)
        df_final = df_final.drop_duplicates(path, keep = "first")
        df_last_matches.to_csv(path, index = False)

    except:
        df_last_matches.to_csv(path, index = False)

