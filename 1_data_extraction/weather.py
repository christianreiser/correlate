import os
from datetime import datetime

import pandas as pd

from helper import histograms

"""
go through csv files
create aggregation df for each file
select which aggregation is needed: max, min, mean, sum
append to data frame
write dataframe
"""
path_to_csv_files = '/home/chrei/PycharmProjects/correlate/0_data_raw/weather/'
outputname = '/home/chrei/PycharmProjects/correlate/0_data_raw/weather/weather_daily_summaries.csv'
verbose = False
excludedFiles = ['weather (another copy).csv', 'weather (copy).csv']
print('starting ...')


def csv_2_df(csv_root, csv_file):
    """
    as name says
    """
    feature = os.path.splitext(csv_file)[0]
    if verbose:
        print('feature:', feature)
        print(csv_root, csv_file)
    df = pd.read_csv(os.path.join(csv_root, csv_file))  # read csv to df
    if verbose:
        print('read df:', df)

    df = df.drop(['dt', 'timezone', 'city_name', 'lat', 'lon', 'sea_level', 'grnd_level', 'weather_id', 'weather_main',
                  'weather_description', 'weather_icon'], axis=1)

    print('change date format. takes a while...')

    for i, row in df.iterrows():
        date_time = datetime.strptime(str(row['dt_iso'][:-19]), '%Y-%m-%d')
        df.loc[i, 'dt_iso'] = date_time

    print('aggregating')

    df_mean = df.groupby(df['dt_iso']).mean().round(1)
    df_sum = df.groupby(df['dt_iso']).sum().round(1)
    df_min = df.groupby(df['dt_iso']).min().round(1)
    df_max = df.groupby(df['dt_iso']).max().round(1)

    print('building df')

    daily_aggregation_df = pd.DataFrame()
    daily_aggregation_df['wOutTempMean'] = df_mean['temp']
    daily_aggregation_df['wOutTempMin'] = df_min['temp_min']
    daily_aggregation_df['wOutTempMax'] = df_max['temp_max']
    daily_aggregation_df['wOutTempDelta'] = df_max['temp_max'] - df_min['temp_min']

    daily_aggregation_df['wOutTempFeelMean'] = df_mean['feels_like']
    daily_aggregation_df['wOutTempFeelMin'] = df_min['feels_like']
    daily_aggregation_df['wOutTempFeelMax'] = df_max['feels_like']
    daily_aggregation_df['wOutPressMean'] = df_mean['pressure']
    daily_aggregation_df['wOutPressMin'] = df_min['pressure']
    daily_aggregation_df['wOutPressMax'] = df_max['pressure']
    daily_aggregation_df['wOutPressDelta'] = df_max['pressure'] - df_min['pressure']
    daily_aggregation_df['wOutHumMean'] = df_mean['humidity']
    daily_aggregation_df['wOutHumMin'] = df_min['humidity']
    daily_aggregation_df['wOutHumMax'] = df_max['humidity']
    daily_aggregation_df['wOutWindMean'] = df_mean['wind_speed']
    daily_aggregation_df['wOutWindMin'] = df_min['wind_speed']
    daily_aggregation_df['wOutWindMax'] = df_max['wind_speed']
    daily_aggregation_df['wOutCloudMean'] = df_mean['clouds_all']
    daily_aggregation_df['wOutCloudMin'] = df_min['clouds_all']
    daily_aggregation_df['wOutCloudMax'] = df_max['clouds_all']
    daily_aggregation_df['wOutPrecipit'] = df_sum['rain_1h'] + df_sum['rain_3h'] + df_sum['snow_1h'] + df_sum[
        'snow_3h']

    return daily_aggregation_df


for root, dirs, files in os.walk(path_to_csv_files):
    for file in files:  # go through all csv files
        if file.endswith("weather.csv") and file.startswith('weather.csv') and file not in excludedFiles:
            df = csv_2_df(root, file)
            # histograms(df, '/home/chrei/PycharmProjects/correlate/plots/raw_distributions/')

# print file
print('writing...')
df.to_csv(outputname)
print('done! :)')
