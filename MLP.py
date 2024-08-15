
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, Normalizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import lightgbm as lgb
import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
#from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from pandas.plotting import scatter_matrix
from datetime import datetime,timedelta

#%%
df = pd.read_csv("outputs\df_for_model.csv")
df_new = pd.read_csv("outputs\master.csv")
df, df_new

#%%
df["datetime"] = df["datetime"].str.replace(r"\+00:00", "", regex=True)
df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")

df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["year"] = df["datetime"].dt.year
df["weekday"] = df["datetime"].dt.weekday
df[["hour", "year", "weekday"]]
#%%
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.date
#%%
def public(x):
    if x < datetime.strptime("2020-05-01", '%Y-%m-%d').date():
        return 1
    else: return 0

df["public"] = df["date"].apply(public)
df["public"]  
#%%
selected_columns = pd.read_csv("outputs\selected_columns_covid.csv")
selected_columns = selected_columns["selected_columns"].values
selected_columns
#%%
for i in df.columns:
    print(i)
#%%
columns_21 = ["date", "team", "first_goals", "match_goals", "shots", "passes_accurate", "red_cards", "yellow_cards", "goalkeeper_saves"]
columns_home_21 = ["date", "home_team", "first_home_goals", "match_home_goals", "total_shots_home", "passes_accurate_home", "red_cards_home", "yellow_cards_home", "goalkeeper_saves_home"]
columns_away_21 = ["date", "away_team", "first_away_goals", "match_away_goals", "total_shots_away", "passes_accurate_away", "red_cards_away", "yellow_cards_away", "goalkeeper_saves_away"]

df_teams_21 = pd.DataFrame([], columns = columns_21)
for index, row in df.iterrows():
    home_values = row[columns_home_21]
    away_values = row[columns_away_21]
    values = [list(home_values.values)] + [list(away_values.values)]
    df_temp = pd.DataFrame(values, columns = columns_21)
    
    df_teams_21 = pd.concat([df_teams_21, df_temp], axis = 0)

df_teams_21 = df_teams_21.reset_index(drop = True)
print(df_teams_21)
#%%
df_teams_21.columns

#%%
#Stats of last 21 days
def get_matches_21(x):
    home_team = x["home_team"]
    away_team = x["away_team"]
    date = x["date"]
    
    df_temp = df_teams_21.loc[(df_teams_21["date"] >= (date - timedelta(21))) & (df_teams_21["date"] < date)]
    df_home = df_temp.loc[(df_temp["team"] == home_team)]
    df_away = df_temp.loc[(df_temp["team"] == away_team)]
    
    home_matches = len(df_home)
    away_matches = len(df_away)
    try:
        comp_matches = away_matches / home_matches
    except:
        comp_matches = home_matches
    
    first_home_goals = df_home["first_goals"].sum()
    first_away_goals = df_away["first_goals"].sum()
    try:
        comp_first_goals = first_away_goals / first_home_goals
    except:
        comp_first_goals = 0
        
    match_home_goals = df_home["match_goals"].sum()
    match_away_goals = df_away["match_goals"].sum()
    try:
        comp_match_goals = match_away_goals / match_home_goals
    except:
        comp_match_goals = 0
        
    home_shots = df_home["shots"].sum()
    away_shots = df_away["shots"].sum()
    try:
        comp_shots = away_shots / home_shots
    except:
        comp_shots = 0   
        
    home_passes_accurate = df_home["passes_accurate"].sum()
    away_passes_accurate = df_away["passes_accurate"].sum()
    try:
        comp_passes_accurate = away_passes_accurate / home_passes_accurate
    except:
        comp_passes_accurate = 0
        
    home_red_cards = df_home["red_cards"].sum()
    away_red_cards = df_away["red_cards"].sum()
    try:
        comp_red_cards = away_red_cards / home_red_cards
    except:
        comp_red_cards = 0
    
    home_yellow_cards = df_home["yellow_cards"].sum()
    away_yellow_cards = df_away["yellow_cards"].sum()
    try:
        comp_yellow_cards = away_yellow_cards / home_yellow_cards
    except:
        comp_yellow_cards = 0
    
    home_goalkeeper_saves = df_home["goalkeeper_saves"].sum()
    away_goalkeeper_saves = df_away["goalkeeper_saves"].sum()
    try:
        comp_goalkeeper_saves = away_goalkeeper_saves / home_goalkeeper_saves
    except:
        comp_goalkeeper_saves = 0
    
    return comp_matches, comp_first_goals, comp_match_goals, comp_shots, comp_passes_accurate, comp_red_cards, comp_yellow_cards, comp_goalkeeper_saves

df["comp_matches"], df["comp_first_goals"], df["comp_match_goals"], df["comp_shots"], df["comp_passes_accurate"], df["comp_red_cards"], df["comp_yellow_cards"], df["comp_goalkeeper_saves"] = zip(*df.apply(get_matches_21, axis = 1))
df[["comp_matches", "comp_first_goals", "comp_match_goals", "comp_shots", "comp_passes_accurate", "comp_red_cards", "comp_yellow_cards", "comp_goalkeeper_saves"]].describe()
#%%
df
#%%
selected_columns = [
       #'home_team','away_team',
        
       'league',
       'round',
       #'last_h2h_total', 'last_h2h_home_won_total',
       #'last_h2h_away_won_total', 'last_h2h_draw_away_total',
       #'comparison_forme_home',
       #'comparison_forme_away',
       #'comparison_att_home', 'comparison_att_away',
       #'comparison_def_home', 'comparison_def_away',
       #'comparison_fishlaw_home', 'comparison_fishlaw_away',
       #'comparison_h2h_home', 'comparison_h2h_away',
       #'comparison_goalsh2h_home', 'comparison_goalsh2h_away',
       #'advice_model',
       #'public',
       
       #"home_pred", "draw_pred", "away_pred",
       
       "comp_matches", "comp_match_goals",
       "comp_shots", "comp_passes_accurate", 
       "comp_red_cards",
       "comp_yellow_cards", "comp_goalkeeper_saves", "comp_first_goals",
       'last_h2h_home_home_total/last_h2h_away_home_total',
       'last_h2h_home_won_home/last_h2h_home_won_away',
       'last_h2h_home_won_total/last_h2h_away_won_total',
       'last_h2h_draw_home_home/last_h2h_draw_away_home',
       'comparison_forme_home/comparison_forme_away',
       'comparison_att_home/comparison_att_away',
       'comparison_def_home/comparison_def_away',
       'comparison_fishlaw_home/comparison_fishlaw_away',
       'comparison_h2h_home/comparison_h2h_away',
       'comparison_goalsh2h_home/comparison_goalsh2h_away',
       
       "cum_goalkeeper_saves_home_total",
       "cum_offsides_home_total",
       "cum_passes_%_home_avg",
       "cum_passes_accurate_home_total",
       "cum_total_shots_home_total",
       
       "cum_goalkeeper_saves_away_total",
       "cum_offsides_away_total",
       "cum_passes_%_away_avg",
       "cum_passes_accurate_away_total",
       "cum_total_shots_away_total",
       
       
       
]

selected_columns
#%%
cum_columns = [x for x in df.columns if ("cum_" in x) and ("/" not in x)]
division_columns = [x for x in df.columns if ("/" in x)]
odd_columns = [x for x in df_new.columns if ("odd_" in x)]
print(division_columns)
#selected_columns = [x for x in selected_columns if x not in division_columns]
print(cum_columns)

#%%
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('outliers', RobustScaler()),
    ('normalizer', Normalizer())])
onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
#%%
def duplicate_away_winner(dff):
    
    df_away_winner = dff.loc[dff["winner"] == 2]
    
    dff = pd.concat([dff, df_away_winner], axis = 0)
    return dff

def multiply_draw(dff, frac):
    
    df_draw = dff.loc[dff["winner"] == 1]
    df_draw = df_draw.sample(frac=frac, random_state=1)
    
    dff = pd.concat([dff, df_draw], axis = 0)
    return dff

#%%
df_model = df.sample(frac=1, random_state=1)

df_model = df_model.loc[df_model[cum_columns].notnull().all(axis=1)]
#df_model = df_model.loc[df_model["public"]==1]
print(len(df_model))
#df_model = df_model[selected_columns + ["winner"]]
#df_model = df_model.loc[df_model["league"].isin(['Primeira Liga'])]

teams_list = ["Atletico Madrid", "Mallorca", "Norwich", "Brighton", "Chelsea", "Watford",
              "Granada CF", "Valencia", "Southampton", "Manchester City",
              "Real Madrid", "Athletic Club", "Inter", "Bologna",
              "Cagliari", "Atalanta", "Sampdoria", "Spal"]
df_model = df_model.loc[df_model["home_team"].isin(teams_list) | 
                        df_model["away_team"].isin(teams_list)]


df_model = df_model.replace(-np.inf, 0)
df_model = df_model.replace(np.inf, 0)
#df_model = df_model.replace(np.nan, 0)
#print(df_model.isna().sum())

df_model = df_model.reset_index(drop = True)

#X_validation = df_model.loc[df_model["fixture_id"].isin(df_new["fixture_id"])]
#X_model = df_model.loc[~df_model["fixture_id"].isin(df_new["fixture_id"])]
#X_model = df_model.loc[df_model["year"] == 2020]
#X_validation = df_model.loc[df_model["fixture_id"].isin(df_new["fixture_id"])]
#X_model = df_model.loc[(df_model["month"].isin([5,6,7]))]
#X_model = df_model.loc[(df_model["year"] == 2020)]

#X_model = df_model.loc[(df_model["season"] == 2019)]
#X_model = duplicate_no_home_winner(X_model)

X_validation = df_model.loc[df_model["public"]==0]
X_model = df_model.loc[(df_model["season"].isin([2019]))]
X_model = X_model.loc[(X_model["public"] == 1)]
X_model = duplicate_away_winner(X_model)
X_model = multiply_draw(X_model, 0.25)
#X_model = X_model.loc[~X_model["fixture_id"].isin(df_new["fixture_id"])]

#X_model, X_validation = train_test_split(df_model, test_size=.1,
#                                        stratify=df_model["winner"], random_state=1)

#X_model = fix_duplicating(X_model, "winner")
X = X_model[selected_columns]
y = X_model["winner"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y,random_state=1)

print("val, train, test: ", len(X_validation), len(X_train), len(X_test))
print(df_model.home_team.unique())
print(df_model.away_team.unique())
#%%
[x for x in teams_list
 if (x not in list(df_model.home_team.unique()) + list(df_model.home_team.unique()))]
#%%
for i in X.columns:
    na = X[i].isna().sum()
    print(i,na)
len(X)

#%%
print(f'Train datasets: X: {X_train.shape}, Y: {y_train.shape}')
print(f'Test datasets: X: {X_test.shape}, Y: {y_test.shape}')

numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'float32', 'int32']).columns
onehot_vars = X_train.select_dtypes(include=['object']).columns

print("numeric features: ",numeric_features)
print("one hot features: ",onehot_vars)


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('onehot_cat', onehot_transformer, onehot_vars)
        ]
    )

#Final pipe
pipe_lgbm = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(lgb.LGBMClassifier(n_jobs=3), threshold='1.5*mean')),
        #('balancer', SMOTE()),
        ('classifier', lgb.LGBMClassifier(n_jobs=6))
        ])

pipe_mlp = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(lgb.LGBMClassifier(n_jobs=3), threshold='1.5*mean')),
        ('balancer', SMOTE(random_state=1)),
        ('classifier', MLPClassifier())
        ])


# Specify parameters and distributions to sample from (https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761)

param_dist_lgbm = {'classifier__num_leaves': sp_randint(2,8,20),
             'classifier__min_child_samples': sp_randint(2,8,20),
             'classifier__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1,2,5],
             'classifier__subsample': sp_uniform(loc=0.01, scale=1),
             'classifier__colsample_bytree': sp_uniform(loc=0.1, scale=2),
             'classifier__reg_alpha': [0, 1e-1, 1, 2, 5, 10,20],
             'classifier__reg_lambda': [0, 1e-1, 1, 2, 5, 10,20],
             'classifier__learning_rate': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5,10],
             'classifier__n_estimators': [3,4,5, 10, 15, 20,50]
             }

param_dist_mlp = {
    'classifier__hidden_layer_sizes': [(50,25,50),(50, 100, 50), (50,100,25),(25,50,10)],
#    'classifier__hidden_layer_sizes': [(50,100,25), (250, 500, 100), (500, 500, 250, 100), (100, 500, 100), (500, 250, 100)],
    'classifier__activation': ['tanh', 'relu', 'identity', 'logistic'],
    'classifier__solver': ['sgd', 'adam'],
    'classifier__alpha': [0.001, 0.01, 0.1, 0.25],
    'classifier__learning_rate': ['constant','adaptive'],
    'classifier__max_iter': [100, 200, 350, 500]
    }
#(50, 100, 50), (50,100,25),(25,50,10), (50,25,50)
#param_dist_mlp = {
#    'classifier__hidden_layer_sizes': [(50, 100, 25)],
#    'classifier__activation': ['tanh'],
#    #'classifier__solver': ['sgd'],
#    'classifier__alpha': [0.001, 0.002, 0.0025],
#    #'classifier__learning_rate': ['adaptive'],
#    'classifier__max_iter': [350]
#    }
param_dist_mlp = {
    'classifier__hidden_layer_sizes': [(20,50,20),(10, 20, 10), (25,50,25),(15,20,15)],
#    'classifier__hidden_layer_sizes': [(50,100,25), (250, 500, 100), (500, 500, 250, 100), (100, 500, 100), (500, 250, 100)],
    'classifier__activation': ['tanh', 'relu', 'identity', 'logistic'],
    'classifier__solver': ['sgd', 'adam'],
    'classifier__alpha': [0.001, 0.01, 0.1, 0.25],
    'classifier__learning_rate': ['constant','adaptive'],
    'classifier__max_iter': [50, 100, 150]
    }

n_models = 10
LGBM = RandomizedSearchCV(pipe_lgbm, param_distributions=param_dist_lgbm, n_iter=n_models, n_jobs=3,
                              scoring='balanced_accuracy', cv=5, verbose=1, random_state=1)

MLP = RandomizedSearchCV(pipe_mlp, param_distributions=param_dist_mlp, n_iter=n_models, n_jobs=6,
                              scoring='balanced_accuracy', cv=5, verbose=1, random_state=1)
print("Start training")
MLP.fit(X_train, y_train)
MLP = MLP.best_estimator_

y_train_pred = MLP.predict(X_train)
y_test_pred = MLP.predict(X_test)
y_val_pred = MLP.predict(X_validation[selected_columns])

print('TRAIN accuracy: ', accuracy_score(y_train, y_train_pred))
print('TEST accuracy: ', accuracy_score(y_test, y_test_pred))
print('VALIDATION accuracy: ', accuracy_score(X_validation["winner"], y_val_pred))

print('TRAIN F1 score: ', f1_score(y_train, y_train_pred,average = "weighted"))
print('TEST F1 score: ', f1_score(y_test, y_test_pred,average = "weighted"))
print('VALIDATION F1 score: ', f1_score(X_validation["winner"], y_val_pred,average = "weighted"))

train_cm_norm = confusion_matrix(y_train, y_train_pred, normalize = "true")
train_cm = confusion_matrix(y_train, y_train_pred)
print('TRAIN Confusion Matrix')
print(train_cm_norm)
print(train_cm)
test_cm_norm = confusion_matrix(y_test, y_test_pred, normalize = "true")
test_cm = confusion_matrix(y_test, y_test_pred)
print('TEST Confusion Matrix')
print(test_cm_norm)
print(test_cm)
val_cm_norm = confusion_matrix(X_validation["winner"], y_val_pred, normalize = "true")
val_cm = confusion_matrix(X_validation["winner"], y_val_pred)
print('VALIDATION Confusion Matrix')
print(val_cm_norm)
print(val_cm)

#%%
MLP["classifier"]
#%%
MLP_old = pk.load(open("outputs/MLP_v4.pk", 'rb'))
MLP_old["classifier"]

#%%
pk.dump(MLP, open("outputs\MLP_v6.pk", 'wb'))

df_selected_columns_ex6 = pd.DataFrame(selected_columns, columns = ["selected_columns"]) 
df_selected_columns_ex6.to_csv("selected_columns_ex6.csv", index = False)

#%%

df_new_cruce = df_new[odd_columns + ["fixture_id"]]
df_new_cruce.columns
#%%
df_validation = df_model.loc[df_model["fixture_id"].isin(df_new["fixture_id"])]
df_validation = df_validation.merge(df_new_cruce, on = "fixture_id", how = "left")
df_validation["PRED_MLP"] = MLP.predict(df_validation[selected_columns])
df_validation["PROB_L_MLP"] = MLP.predict_proba(df_validation[selected_columns])[:,0]
df_validation["PROB_V_MLP"] = MLP.predict_proba(df_validation[selected_columns])[:,2]
df_validation["PROB_E_MLP"] = MLP.predict_proba(df_validation[selected_columns])[:,1]
#%%
df_validation.to_csv("results/final_pred.csv", index = False)
#%%

