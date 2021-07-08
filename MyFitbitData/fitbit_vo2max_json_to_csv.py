import os
from datetime import datetime, timedelta

import pandas as pd

path_to_json_files = '/home/chrei/PycharmProjects/correlate/MyFitbitData/ChrisRe/Physical Activity'
output_filename = 'fitbit_vo2max.csv'
verbose = True

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


coldStart = True
for root, dirs, files in os.walk(path_to_json_files):
    for file in files:  # go through all json files
        # take only .json files, exclude averages.json because of different format and
        # exclude files small files to reduce noise
        # exclude correlations.json
        file_size = os.stat(os.path.join(path_to_json_files, file))[6]
        if file.endswith(".json") and file.startswith('exercise-') and file not in excludedFiles:

            df = pd.read_json(os.path.join(root, file))  # read json to d
            # df = df.set_index('originalStartTime')  # set date as index

            # all_files_df = pd.DataFrame(columns=['originalStartTime', 'vo2Max', 'originalDuration'])
            if 'vo2Max' in df:
                df = df.dropna()
                df_target = df['originalStartTime']
                # test = df['vo2Max']
                # vo2Max = [d.get('vo2Max') for d in df['vo2Max'].dic]

                df_target = pd.concat([df_target, df['vo2Max']], axis=1)
                # df_target = pd.concat([df_target, df['originalDuration']], axis=1)
                if coldStart:
                    all_files_df = df_target
                    coldStart = False
                else:
                    all_files_df = all_files_df.append(df_target)

# extract from dict
all_files_df['vo2max'] = [d.get('vo2Max') for d in all_files_df['vo2Max']]
all_files_df = all_files_df.drop('vo2Max', axis=1)

# aggregate when multiple per day
all_files_df['originalStartTime'] = pd.to_datetime(all_files_df['originalStartTime'])
all_files_df = all_files_df.groupby(all_files_df['originalStartTime'].dt.date).mean()

# round
all_files_df.vo2max = round(all_files_df.vo2max, 0)

# fill missing dates with NaN
all_files_df.reset_index(level=0, inplace=True)
firstDate = all_files_df['originalStartTime'][0]
lastDate = all_files_df['originalStartTime'].iloc[-1]
all_files_df = all_files_df.set_index('originalStartTime')  # set date as index
idx = pd.date_range(firstDate, lastDate)
all_files_df.index = pd.DatetimeIndex(all_files_df.index)
all_files_df = all_files_df.reindex(idx, fill_value=float("NaN"))


# write file
all_files_df.to_csv(output_filename)
if verbose:
    print(all_files_df)
print(str(output_filename) + '.csv written')
