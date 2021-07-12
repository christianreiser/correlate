import os
from datetime import datetime

import pandas as pd

"""
go through csv files
create aggregation df for each file
select which aggregation is needed: max, min, mean, sum
append to data frame
write dataframe
"""
path_to_csv_files = '/home/chrei/PycharmProjects/correlate/0_data_raw/weather/'
outputname = 'weather_extracted.csv'
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

    df_mean = df.groupby(df['dt_iso']).mean()
    df_sum = df.groupby(df['dt_iso']).sum()
    df_min = df.groupby(df['dt_iso']).min()
    df_max = df.groupby(df['dt_iso']).max()

    print('building df')

    daily_aggregation_df = pd.DataFrame()
    daily_aggregation_df['w_temp_mean'] = df_mean['temp']
    daily_aggregation_df['w_temp_min'] = df_min['temp_min']
    daily_aggregation_df['w_temp_max'] = df_max['temp_max']
    daily_aggregation_df['w_temp_delta'] = df_max['temp_max'] - df_min['temp_min']

    daily_aggregation_df['w_temp_feels_mean'] = df_mean['feels_like']
    daily_aggregation_df['w_temp_feels_min'] = df_min['feels_like']
    daily_aggregation_df['w_temp_feels_max'] = df_max['feels_like']
    daily_aggregation_df['w_press_mean'] = df_mean['pressure']
    daily_aggregation_df['w_press_min'] = df_min['pressure']
    daily_aggregation_df['w_press_max'] = df_max['pressure']
    daily_aggregation_df['w_press_delta'] = df_max['pressure'] - df_min['pressure']
    daily_aggregation_df['w_hum_mean'] = df_mean['humidity']
    daily_aggregation_df['w_hum_min'] = df_min['humidity']
    daily_aggregation_df['w_hum_max'] = df_max['humidity']
    daily_aggregation_df['w_wind_mean'] = df_mean['wind_speed']
    daily_aggregation_df['w_wind_min'] = df_min['wind_speed']
    daily_aggregation_df['w_wind_max'] = df_max['wind_speed']
    daily_aggregation_df['w_cloud_mean'] = df_mean['clouds_all']
    daily_aggregation_df['w_cloud_min'] = df_min['clouds_all']
    daily_aggregation_df['w_cloud_max'] = df_max['clouds_all']
    daily_aggregation_df['w_precipitation'] = df_sum['rain_1h'] + df_sum['rain_3h'] + df_sum['snow_1h'] + df_sum['snow_3h']

    return daily_aggregation_df


for root, dirs, files in os.walk(path_to_csv_files):
    for file in files:  # go through all csv files
        if file.endswith(".csv") and file.startswith('weather') and file not in excludedFiles:
            df = csv_2_df(root, file)

# print file
print('writing...')
df.to_csv(outputname)
print('done! :)')
