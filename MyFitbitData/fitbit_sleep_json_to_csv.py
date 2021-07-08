import math
import os
from datetime import datetime, timedelta

import pandas as pd

path_to_json_files = '/home/chrei/PycharmProjects/correlate/MyFitbitData/ChrisRe/Sleep'
output_filename = 'fitbit_sleep.csv'
verbose = False

excludedFiles = ['']

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
    df = df.drop(['type', 'infoCode'], axis=1)
    df = df.set_index('dateOfSleep')  # set date as index

    start_min_before_midnight = []
    end_min_after_midnight = []
    duration_min = []
    deep = []
    wake = []
    light = []
    rem = []
    efficiency = []
    nap_detected = []
    for i, row in df.iterrows():

        if verbose:
            print("row['startTime']", row['startTime'])

        # set nap sleep start and end to 0 so it doesn't affect sleep time after reduce sum
        if row['mainSleep']:
            start_time = datetime.strptime(row['startTime'], '%Y-%m-%dT%H:%M:%S.%f')
            midnight_time = round_time(start_time, roundTo=24 * 60 * 60)  # round to midnight
            start_min_before_midnight.append(math.floor((midnight_time - start_time).total_seconds() / 60))
            end_time = datetime.strptime(row['endTime'], '%Y-%m-%dT%H:%M:%S.%f')
            midnight_time = round_time(end_time, roundTo=24 * 60 * 60)  # round to midnight
            end_min_after_midnight.append(math.floor((end_time - midnight_time).total_seconds() / 60))
            efficiency.append(row['efficiency'])
            nap_detected.append(0)
        elif not row['mainSleep']:
            start_min_before_midnight.append(0)
            end_min_after_midnight.append(0)
            efficiency.append(0)
            nap_detected.append(1)

        duration_min.append(math.floor(round(row['duration'] / 60000, 0)))
        summary = row['levels']['summary']
        if [*summary] == ['deep', 'wake', 'light', 'rem']:
            deep.append(summary['deep']['minutes'])
            wake.append(summary['wake']['minutes'])
            light.append(summary['light']['minutes'])
            rem.append(summary['rem']['minutes'])
        else:
            deep.append(float("NaN"))
            wake.append(float("NaN"))
            light.append(float("NaN"))
            rem.append(float("NaN"))

    # add new columns
    df['startBeforeMidnight'] = start_min_before_midnight
    df['endBeforeMidnight'] = end_min_after_midnight
    df['duration'] = duration_min
    df['deep'] = deep
    df['wake'] = wake
    df['light'] = light
    df['rem'] = rem
    df['m_efficiency'] = efficiency
    df['nap_detected'] = nap_detected

    # remove old replaced columns
    df = df.drop(['startTime', 'endTime', 'duration', 'levels', 'efficiency'], axis=1)  # ,'mainSleep'

    if verbose:
        print('df', df)
    return df


# create df from json files
if verbose:
    print('path', path_to_json_files)

for root, dirs, files in os.walk(path_to_json_files):
    coldStart = True
    for file in files:  # go through all json files
        if verbose:
            print('Filename=', file)

        # take only .json files, exclude averages.json because of different format and
        # exclude files small files to reduce noise
        # exclude correlations.json
        file_size = os.stat(os.path.join(path_to_json_files, file))[6]
        if file.endswith(".json") and file.startswith('sleep-') and file not in excludedFiles:
            if verbose:
                print('Filename target =', file)
                print('file-size=', file_size)
                print('json_2_1_feature_df(root, file):', json_to_df(root, file))
            if coldStart:
                all_files_df = json_to_df(root, file)
            else:
                all_files_df = all_files_df.append(json_to_df(root, file))
            coldStart = False

# sort by date
all_files_df = all_files_df.sort_values(by='dateOfSleep')

# convert-index-of-a-pandas-dataframe-into-a-column
all_files_df.reset_index(level=0, inplace=True)

# drop duplicates
all_files_df = all_files_df.drop_duplicates(subset="logId")

# # aggregate when multiple sleep times per day
all_files_df['dateOfSleep'] = pd.to_datetime(all_files_df['dateOfSleep'])
all_files_df = all_files_df.groupby(all_files_df['dateOfSleep'].dt.date).sum()

all_files_df = all_files_df.drop(['logId', 'mainSleep', 'minutesAwake'], axis=1)

all_files_df.to_csv(output_filename)
if verbose:
    print(all_files_df)
print(str(output_filename) + '.csv written')
