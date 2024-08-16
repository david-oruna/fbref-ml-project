# Football Match Prediction Model 🏆⚽

## Overview

This project provides secure and reliable football match predictions by combining data from `api-football` predictions and a trained Machine Learning (ML) model. The process involves several stages: data extraction, data cleaning and wrangling, model training, and generating predictions based on new match fixtures.

## Project Structure

- **Data Extraction**
  - The match fixtures, statistics, and predictions are extracted from the `api-football` API.
  
- **Data Cleaning and Wrangling**
  - The raw data extracted from the API is cleaned and formatted using `model_data_wrangling.py`. This script processes dates, generates cumulative statistics, and creates additional features that are used for model training.

- **Model Training**
  - The cleaned and processed data is then used to train an ML model using the script `MLP.py`. The training process involves data preprocessing, feature engineering, and model selection using techniques like RandomizedSearchCV. The trained model is saved as a `.pk` file for later use.

- **Generating Predictions**
  - The final step is generating predictions for individual fixtures. The user inputs a fixture ID, and the trained model predicts the match outcome (home win, draw, or away win).


