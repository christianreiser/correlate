import json
import pandas as pd
import matplotlib.pyplot as plt


"""
correlate steps and min active

load data form json, into df, correlation matrix, visualize
"""
value_steps_active_min = []
date_steps_active_min = []
value_steps = []
date_steps = []

with open('./exist0/steps.json') as json_file:
    steps_loaded = json.load(json_file)
    for entry in steps_loaded:
        value_steps.append(entry['value'])
        date_steps.append(entry['date'])

with open('./exist0/steps_active_min.json') as json_file:
    steps_loaded = json.load(json_file)
    for entry in steps_loaded:
        value_steps_active_min.append(entry['value'])
        date_steps_active_min.append(entry['date'])

#plt.plot(steps, steps_active_min, 'ro')
#plt.show()

df_steps = pd.DataFrame(value_steps, date_steps)
df_steps_active_min = pd.DataFrame(value_steps_active_min, date_steps_active_min)
#print(df_steps)
#print(df_steps_active_min)

#def json_2_1_feature_df(feature, json_file_path):
"""
"""
dftest = pd.read_json('./exist0/steps.json')  # read json to df
dftest = dftest.rename(columns={"value": "steps"})  # rename columnname value to steps
dftest = dftest.set_index('date')  # set date as index
print(dftest)

#json_2_1_feature_df(steps, './exist0/steps.json'
