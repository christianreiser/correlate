import os
from datetime import datetime, timedelta
import math

import pandas as pd

"""
correlate

load data form json, into df, correlation matrix, visualize
"""
path_to_json_files = '/home/chrei/PycharmProjects/correlate/MyFitbitData/ChrisRe/Sleep'
output_filename = 'exist_output_2019.csv'
verbose = True

excludedFiles = ['averages.json', 'correlations.json', 'weather_summary.json', 'twitter_username.json',
                 'weather_icon.json', 'mood_note.json', 'custom.json', 'location_name.json']

if verbose:
    print('start running...')


def round_time(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt == None: dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def json_to_df(json_root, json_file):
    """
    as name says
    """
    feature = os.path.splitext(json_file)[0]
    if verbose:
        print(json_root, json_file)
    df = pd.read_json(os.path.join(json_root, json_file))  # read json to d
    if verbose:
        print('df[levels]', df['levels'])
    df = df.drop(['logId', 'type', 'infoCode', 'levels'], axis=1)
    df = df.set_index('dateOfSleep')  # set date as index

    for i, row in df.iterrows():
        if verbose:
            print("row['startTime']", row['startTime'])
        start_time = datetime.strptime(row['startTime'], '%Y-%m-%dT%H:%M:%S.%f')
        midnight_time = round_time(start_time, roundTo=24 * 60 * 60) #round to midnight
        start_minutes_before_midnight = math.floor((midnight_time - start_time).total_seconds() / 60)
        end_time = datetime.strptime(row['endTime'], '%Y-%m-%dT%H:%M:%S.%f')
        midnight_time = round_time(end_time, roundTo=24 * 60 * 60) #round to midnight
        end_minutes_after_midnight = math.floor((end_time-midnight_time).total_seconds() / 60)

    if verbose:
        print('df', df)
    return df


# create df from json files
if verbose:
    print('path', path_to_json_files)

for root, dirs, files in os.walk(path_to_json_files):
    i = 0
    for file in files:  # go through all json files
        if verbose:
            print('Filename=', file)

        # take only .json files, exclude averages.json because of different format and
        # exclude files small files to reduce noise
        # exclude correlations.json
        file_size = os.stat(os.path.join(path_to_json_files, file))[6]
        if file.endswith(".json") and file.startswith('sleep-') and file not in excludedFiles:
            if verbose:
                print('file-size=', file_size)
                print('json_2_1_feature_df(root, file):', json_to_df(root, file))
            if i == 0:
                df_0 = json_to_df(root, file)
            elif i == 1:
                df_1 = json_to_df(root, file)
                df = df_0.join(df_1)
            else:
                df_1 = json_to_df(root, file)
                df = df.join(df_1)
                if verbose:
                    print(df)
                    print('/n')
            i += 1

df.to_csv(output_filename)
if verbose:
    print(df)
print('exist_output_2019.csv written')
