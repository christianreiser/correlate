from datetime import datetime

import pandas as pd

target_label = 'mood'

df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)
print(df.columns)

# drop gps location
# df = df.drop(
#     ['Low latitude (deg)', 'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)'], axis=1)

# drop nutrition
df = df.drop(
    ['sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in'], axis=1)

# 'sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in',

# interpolate weight and vo2 max linearly
df['weight'] = df['weight'].interpolate(method='linear')
df['VO2Max'] = df['VO2Max'].interpolate(method='linear')

# drop days where too much data is missing
datelist = pd.date_range(datetime.strptime('2019-02-11', '%Y-%m-%d'), periods=615).tolist()  #
datelist = datelist + pd.date_range(datetime.strptime('2021-06-16', '%Y-%m-%d'), periods=41).tolist()
datelist = [day.strftime('%Y-%m-%d') for day in datelist]
df = df.drop(datelist, axis=0)

# drop days without mood rating
for day, _ in df.iterrows():
    # print('day',day)
    if df[target_label][day] != df[target_label][day]:  # checks for NaN
        df = df.drop(day)

# fill missing values with mean value
# get mean value
# mean = df.agg(['mean'], axis=0)
for attribute_name in df.columns:
    nan_dates = []
    nan_data_true_false = pd.isnull(df[attribute_name])
    nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
    nan_dates = nan_data_true_false[nan_numeric_indices].index
    if len(nan_dates) > 0:
        atest = nan_dates
        print('d')
    # for nan_date in nan_dates:
    # substitute = mean[attribute_name][0]
    # df.at[nan_date, attribute_name] = substitute

print(df.columns)
