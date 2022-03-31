import os
from datetime import datetime, timedelta
import json

import dateutil
import pandas as pd
from math import floor

path_to_json_files = '/home/chrei/PycharmProjects/correlate/0_data_raw/google/takeout-20210625T075514Z-001/Takeout/Fit/All Sessions'
output_filename = 'google_meditation.csv'
verbose = True

excludedFiles = ['']

if verbose:
    print('start running...')


def round_time(dt=None, roundTo=60):
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
        if file.endswith("_MEDITATION.json") and file.startswith('2') and file not in excludedFiles:
            print('file',file)
            with open(os.path.join(root, file)) as json_data:
                data = json.load(json_data)
                data.pop('endTime', None)
                data.pop('segment', None)
                data.pop('aggregate', None)
                data.pop('fitnessActivity', None)
                data['duration_min'] = floor(float(data['duration'][:-1])/60)
                data.pop('duration', None)

                data['startTime'] = dateutil.parser.isoparse(data['startTime'])
            df = pd.DataFrame.from_dict([data])  # read json to d
            if coldStart:
                all_df = df
                coldStart = False
            else:
                all_df = all_df.append(df)






# aggregate when multiple per day
all_df['startTime'] = pd.to_datetime(all_df['startTime'])
all_df = all_df.groupby(all_df['startTime'].dt.date).sum()

# # round
# all_df.filteredRunVO2Max = round(all_df.filteredRunVO2Max, 1)

# fill missing dates with NaN
all_df.reset_index(level=0, inplace=True)
firstDate = datetime.strptime('2019/02/10', '%Y/%m/%d')  # all_df['dateTime'][0]
lastDate = all_df['startTime'].iloc[-1]
all_df = all_df.set_index('startTime')  # set date as index
idx = pd.date_range(firstDate, lastDate)
all_df.index = pd.DatetimeIndex(all_df.index)
all_df = all_df.reindex(idx, fill_value=float("0"))

# write file
all_df.to_csv(output_filename)
if verbose:
    print(all_df)
print(str(output_filename) + '.csv written')
