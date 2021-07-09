import pandas as pd

target_label = 'mood'

df = pd.read_csv('/home/chrei/code/quantifiedSelfData/daily_summaries.csv', index_col=0)
print(df.columns)

# drop gps location
df = df.drop(
    ['Low latitude (deg)', 'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)'], axis=1)

# drop nutrition
df = df.drop(
    ['sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in'], axis=1)

# 'sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in',

# drop days without mood rating
for day, _ in df.iterrows():
    # print('day',day)
    if df[target_label][day] != df[target_label][day]:  # checks for NaN
        df = df.drop(day)

# drop days where too much data is missing
df = df.drop(['2019-02-11', '2019-02-12', '2019-02-12', '2019-02-13'])

# fill missing values with mean value
# get mean value
# mean = df.agg(['mean'], axis=0)
for attribute_name in df.columns:
    nan_dates = []
    nan_data_true_false = pd.isnull(df[attribute_name])
    nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
    nan_dates = nan_data_true_false[nan_numeric_indices].index
    print('d')
    # for nan_date in nan_dates:
    # substitute = mean[attribute_name][0]
    # df.at[nan_date, attribute_name] = substitute

print(df.columns)
