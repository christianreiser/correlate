import json
import pandas as pd
import matplotlib.pyplot as plt


"""
correlate steps and min active

load data form json, into df, correlation matrix, visualize
"""

def json_2_1_feature_df(feature, json_file_path):
    """
    as name says
    """
    df = pd.read_json(json_file_path)  # read json to df
    df = df.rename(columns={"value": feature})  # rename columnname value to steps
    df = df.set_index('date')  # set date as index
    return df

# json_2_1_feature_df
df_steps = json_2_1_feature_df("steps", './exist0/steps.json')
df_steps_active_min = json_2_1_feature_df("steps_active_min", './exist0/steps_active_min.json')

# join dfs
df_joined = df_steps.join(df_steps_active_min)

# correlate
corr_matrix = df_joined.corr()

print(df_joined)
print(corr_matrix)
