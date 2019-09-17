import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
correlate steps and min active

load data form json, into df, correlation matrix, visualize
"""
path_to_json_files = './exist0/'
def json_2_1_feature_df(json_root, json_file):
    """
    as name says
    """
    feature = os.path.splitext(json_file)[0]
    #print(json_root, json_file)
    df = pd.read_json(os.path.join(json_root, json_file))  # read json to df
    df = df.rename(columns={"value": feature})  # rename columnname value to feature
    df = df.set_index('date')  # set date as index
    return df

# create df from json files 
for root, dirs, files in os.walk(path_to_json_files):
    i=0
    for file in files:
        if file.endswith(".json"):
            if i == 0:
                df_0 = json_2_1_feature_df(root, file)
            elif i == 1:
                df_1 = json_2_1_feature_df(root, file)
                df = df_0.join(df_1)
            else:
                df_1 = json_2_1_feature_df(root, file)
                df = df.join(df_1)
            i+=1

# correlate
corr_matrix = df.corr()
# corr_matrix = np.fill_diagonal(corr_matrix.values, -1)  # drop selfcorrelation but doesnt work
# print(corr_matrix)


# sort correlation matrix
s = corr_matrix.unstack()
so = s.sort_values(kind="quicksort")
print(so)


# plot
f = plt.figure(figsize=(19, 15))
plt.matshow(corr_matrix, fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=7)
plt.title('Correlation Matrix', fontsize=12);
plt.show()


