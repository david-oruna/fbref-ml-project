# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 16:57:12 2020

@author: USUARIO
"""
#%%
import pandas as pd
#%%
df = pd.read_csv("outputs\master.csv")
df_master_model = pd.read_csv("outputs\master_model.csv")
df["fixture_id"] = df["fixture_id"].astype(int)
df_master_model["fixture_id"] = df_master_model["fixture_id"].astype(int)
#%%
df_next_matches = pd.read_csv("outputs\next_matches.csv")
df_next_matches["fixture_id"] = df_next_matches["fixture_id"].astype(int)
#%%
#df_next_matches.to_csv("C:\\Users\\USUARIO\\Desktop\\AT\\API_football\\outputs\\next_matches.csv", index = False)
#%%
datetime_columns = ["fixture_id", "datetime", "date", "time", "home_team", "away_team"]
df_datetime_master = pd.concat([df_master_model[datetime_columns], df_next_matches[datetime_columns]], axis = 0)
df_datetime_master = df_datetime_master.drop_duplicates("fixture_id")
df_datetime_master = df_datetime_master.reset_index(drop = True)
df_datetime_master
#%%
df_datetime_master.loc[(df_datetime_master["home_team"] == "Famalicao") &
                       (df_datetime_master["away_team"] == "SC Braga")]
#%%
df = df.drop(["date"], axis = 1)
df = df.merge(df_datetime_master, on = "fixture_id", how = "left")

#%%
unnamed_columns = [x for x in df.columns if ("Unnamed" in x)]
unnamed_columns
#%%
df = df.drop(unnamed_columns, axis = 1)
df
#%%
df["country"] = ""
for col in df.columns:
    print(col)
#%%

def prob_recomendada(x):
    
    if x["PROB_L_MLP"]>0.76:
        return x["PROB_L_MLP"]
    elif x["PROB_V_MLP"]>0.76:
        return x["PROB_V_MLP"]
    elif x["PROB_E_MLP"]>0.76:
        return x["PROB_E_MLP"]
    else:
        if x[["PROB_L_MLP", "PROB_V_MLP", "PROB_E_MLP"]].max() < .45:
            return ""
        
        else:
            if x["PROB_L_MLP"] > x["PROB_V_MLP"]:
                return x["PROB_L_MLP"] + x["PROB_E_MLP"]
            if x["PROB_V_MLP"] > x["PROB_L_MLP"]:
                return x["PROB_V_MLP"] + x["PROB_E_MLP"]
            
df["PROB_REC"] = df.apply(prob_recomendada, axis = 1)

def get_cuota(x):
    
    if x["PROB_L_MLP"]>0.76:
        return x["odd_home_wins"]
    elif x["PROB_V_MLP"]>0.76:
        return x["odd_away_wins"]
    elif x["PROB_E_MLP"]>0.76:
        return x["odd_draw"]
    else:
        if x[["PROB_L_MLP", "PROB_V_MLP", "PROB_E_MLP"]].max() < .45:
            return ""
        
        else:
            if x["PROB_L_MLP"] > x["PROB_V_MLP"]:
                return x["odd_home_or_draw"]
            if x["PROB_V_MLP"] > x["PROB_L_MLP"]:
                return x["odd_draw_or_away"]
    
df["cuota"] = df.apply(get_cuota, axis = 1)

def cuota_comb(x):
    if x["PROB_L_MLP"]>x["PROB_V_MLP"]:
        return x["odd_home_or_draw"]
    else: return x["odd_draw_or_away"]
df["cuota_comb"] = df.apply(cuota_comb, axis = 1) 


df["temperatura"] = df["PROB_COMB"] * df["PROB_COMB"] * df["cuota_comb"]

def recomendacion_texto(x):
  
    if x["PROB_L_MLP"]>0.76:
        return x["home_team"] + " gana a " + x["away_team"]
    elif x["PROB_V_MLP"]>0.76:
        return x["away_team"] + " gana a " + x["home_team"]
    elif x["PROB_E_MLP"]>0.76:
        return x["home_team"] + " empata con " + x["away_team"]
    else:
        if x[["PROB_L_MLP", "PROB_V_MLP", "PROB_E_MLP"]].max() < .45:
            return x["home_team"] + " vs " + x["away_team"] + " indeciso"
        
        else:
            if x["PROB_L_MLP"] > x["PROB_V_MLP"]:
                return x["home_team"] + " gana a " + x["away_team"] + " blindada"
            if x["PROB_V_MLP"] > x["PROB_L_MLP"]:
                return x["away_team"] + " gana a " + x["home_team"] + " blindada"

df["recomendacion_texto"] = df.apply(recomendacion_texto, axis = 1)
df["recomendacion_texto"]

#%%
def recomendacion(x):
  
    if x["PROB_L_MLP"]>0.76:
        return 1
    elif x["PROB_E_MLP"]>0.76:
        return "X"
    elif x["PROB_V_MLP"]>0.76:
        return 2
    else:
        if x[["PROB_L_MLP", "PROB_V_MLP", "PROB_E_MLP"]].max() < .45:
            return "indeciso"
        
        else:
            if x["PROB_L_MLP"] > x["PROB_V_MLP"]:
                return "1X"
            if x["PROB_V_MLP"] > x["PROB_L_MLP"]:
                return "X2"

df["recomendacion"] = df.apply(recomendacion, axis = 1)
df["recomendacion"]

#%%
for col in df.columns:
    print(col)

#%%
df.to_csv("outputs\recommendations.csv", index = False)



