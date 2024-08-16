import requests
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import Normalizer, RobustScaler, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
from model_data_wrangling import preprocess_date, generate_advice_model, assign_winner, classify_goals, create_cumulative_stats, stats_columns, create_division_columns
# Load the trained model
with open('MLP_probability.pk', 'rb') as file:
    model = pk.load(file)

# API headers
headers = {
    'x-rapidapi-host': "api-football-v1.p.rapidapi.com",
    'x-rapidapi-key': "your_api_key_here"
}

# Function to fetch fixture data from the API
def get_fixture_data(fixture_id):
    url = f"https://api-football-v1.p.rapidapi.com/v3/fixtures?id={fixture_id}"
    response = requests.get(url, headers=headers)
    fixture_data = response.json()

    fixture_info = fixture_data["response"][0]
    data = {
        "league": fixture_info['league']['name'],
        "fixture_id": fixture_info['fixture']['id'],
        "datetime": fixture_info['fixture']['date'],
        "home_team": fixture_info['teams']['home']['name'],
        "away_team": fixture_info['teams']['away']['name'],
        "referee": fixture_info['fixture']['referee']
    }

    df_fixture = pd.DataFrame([data])
    df_fixture["datetime"] = pd.to_datetime(df_fixture["datetime"].apply(lambda x: x.replace("T", " ")), utc=True)
    df_fixture["time"] = df_fixture["datetime"].dt.time 
    df_fixture["date"] = df_fixture["datetime"].dt.date
    return df_fixture
# Function to preprocess data
def preprocess_fixture(df):
    df = preprocess_date(df)
    df = generate_advice_model(df)
    df = assign_winner(df)
    df = classify_goals(df)
    
    df = create_cumulative_stats(df, stats_columns)
    df = create_division_columns(df)
    return df

def main():
    fixture_id = input("Enter the fixture ID: ")
    df_fixture = get_fixture_data(fixture_id)
    df_fixture = preprocess_fixture(df_fixture)
    
    X = df_fixture  
    # Predicting the outcome
    prediction = model.predict(X)
    prediction_proba = model.predict_proba(X)
    
    # Displaying the prediction
    outcome = {
        0: "Home Win",
        1: "Draw",
        2: "Away Win"
    }
    
    print(f"Prediction: {outcome[prediction[0]]}")
    print(f"Probabilities: Home Win: {prediction_proba[0][0]:.2f}, Draw: {prediction_proba[0][1]:.2f}, Away Win: {prediction_proba[0][2]:.2f}")

if __name__ == "__main__":
    main()