import pandas as pd
import os

"""
correlate

load data form json, into df, correlation matrix, visualize
"""
path_to_json_files = '/home/chrei/code/quantifiedSelfData/2022/ChrisG_a7bdb31dddb7586bea95f752ca7883740c77ae2dde615633ca99b40c74ef9192'
verbose = False

excludedFiles = ['averages.json', 'correlations.json', 'weather_summary.json', 'twitter_username.json',
                 'weather_icon.json', 'mood_note.json', 'custom.json', 'location_name.json']

if verbose:
    print('start running...')


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
if verbose:
    print('path', (path_to_json_files))

for root, dirs, files in os.walk(path_to_json_files):
    i = 0
    for file in files:  # go through all json files
        if verbose:
            print('Filename=', file)

        # take only .json files, exclude averages.json because of different format and
        # exclude files small files to reduce noise
        # exclude correlations.json
        file_size = os.stat(os.path.join(path_to_json_files, file))[6]
        if file.endswith("_2022.json") and file.startswith('data_') and file not in excludedFiles:
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

df.to_csv('exist_output_2022.csv')
if verbose:
    print(df)
print('exist_output_2022.csv written')

