import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np  # needed if self-correlation suppressed

"""
correlate

load data form json, into df, correlation matrix, visualize
"""
path_to_json_files = './ChrisG_dd3d35c4fd41e0386e17b86906d710f6ad86e95c36de734bb4bf7c915d195e8e/'
verbose =  True
compute_correlations = False
min_num_entries = 20  # skipped if smaller

excludedFiles = ['averages.json','correlations.json','weather_summary.json', 'twitter_username.json','weather_icon.json','mood_note.json','custom.json','location_name.json']

def json_2_1_feature_df(json_root, json_file):
    """
    as name says
    """
    feature = os.path.splitext(json_file)[0]
    if verbose:
        print(json_root, json_file)
    df = pd.read_json(os.path.join(json_root, json_file))  # read json to df
    df = df.rename(columns={"value": feature})  # rename column-name value to feature
    df = df.set_index('date')  # set date as index
    return df


# create df from json files 
for root, dirs, files in os.walk(path_to_json_files):
    i = 0
    for file in files:  # go through all json files
        if verbose:
            print('Filename=', file)

        # take only .json files, exclude averages.json because of different format and
        # exclude files small files to reduce noise
        # exclude correlations.json
        file_size = os.stat(os.path.join(path_to_json_files, file))[6]
        # TODO: include averages but has different format
        if file.endswith("_2020.json") and file.startswith(data_)and file not in excludedFiles:
            if verbose:
                print('file-size=', file_size)
                print('json_2_1_feature_df(root, file):', json_2_1_feature_df(root, file))
            if i == 0:
                df_0 = json_2_1_feature_df(root, file)
            elif i == 1:
                df_1 = json_2_1_feature_df(root, file)
                df = df_0.join(df_1)
            else:
                df_1 = json_2_1_feature_df(root, file)
                df = df.join(df_1)
                if verbose:
                    print(df)
                    print('/n')
            i += 1

df.to_csv('exist_output.csv')
if verbose:
    print(df)

# correlate
if compute_correlations:
    corr_matrix = pd.DataFrame.corr(df, method='pearson', min_periods=min_num_entries)
    #corr_matrix = np.fill_diagonal(corr_matrix.values, -1)  # drop self-correlation but doesnt work
    #print(corr_matrix)

    # sort correlation matrix
    s = corr_matrix.unstack()
    so = s.sort_values(kind="quicksort")
    if verbose:
        print(so)

    # plot
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=7)
    plt.title('Correlation Matrix', fontsize=12)
    plt.show()
