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

#Get predictions per game
def get_predictions(df_last_matches):
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

    df_predictions = pd.DataFrame(columns = prediction_cols) 

    for index, row in df_last_matches.iterrows():
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


    df_temp = pd.DataFrame(data,
                            columns = prediction_cols) 
    df_predictions = df_predictions._append(df_temp)
 


    df_predictions = df_predictions.reset_index(drop=True)
    df_predictions
    df_predictions.to_csv("predictions.csv", index = False)

