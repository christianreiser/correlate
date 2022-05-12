import os
from datetime import datetime, timedelta

import pandas as pd

path_to_json_files = '/home/chrei/code/quantifiedSelfData/2022/MyFitbitData/ChrisRe/Physical Activity'
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
        if file.endswith(".json") and file.startswith('run_vo2_max-') and file not in excludedFiles:

            df = pd.read_json(os.path.join(root, file))  # read json to d
            if coldStart:
                all_df = df
                coldStart = False
            else:
                all_df = all_df.append(df)

filteredRunVO2MaxList = []
dateList = []
for i, row in all_df.iterrows():
    filteredRunVO2MaxList.append(round(row.value['filteredRunVO2Max'], 5))
    dateList.append(row.dateTime)

all_files_df = pd.DataFrame(dateList, columns=['dateTime'])
all_files_df['filteredRunVO2Max'] = filteredRunVO2MaxList

# aggregate when multiple per day
all_files_df['dateTime'] = pd.to_datetime(all_files_df['dateTime'])
all_files_df = all_files_df.groupby(all_files_df['dateTime'].dt.date).median()

# round
all_files_df.filteredRunVO2Max = round(all_files_df.filteredRunVO2Max, 1)

# fill missing dates with NaN
all_files_df.reset_index(level=0, inplace=True)
firstDate = datetime.strptime('2019/02/11', '%Y/%m/%d')  # all_files_df['dateTime'][0]
lastDate = all_files_df['dateTime'].iloc[-1]
all_files_df = all_files_df.set_index('dateTime')  # set date as index
idx = pd.date_range(firstDate, lastDate)
all_files_df.index = pd.DatetimeIndex(all_files_df.index)
all_files_df = all_files_df.reindex(idx, fill_value=float("NaN"))

# write file
all_files_df.to_csv(output_filename)
if verbose:
    print(all_files_df)
print(str(output_filename) + '.csv written')
