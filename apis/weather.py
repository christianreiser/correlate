import json
import urllib.request

import numpy as np
import pandas as pd


def flatten_data(y):
    """flatten json"""
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out








def get_one_h_weather_data(long, lat, key_open_weather):
    open_weather_map_request_url = 'https://api.openweathermap.org/data/2.5/weather?lat=' + long + '&lon=' + lat + '&appid=' + key_open_weather

    # contents = urllib.request.urlopen(
    #     "https://api.weatherapi.com/v1/history.json?key=" + key + "&q=" + long + "," + lat + "&dt=" + date).read()
    contents = json.loads(urllib.request.urlopen(open_weather_map_request_url).read())
    contents = flatten_data(contents)

    # match old_weather_headers to dict_keys
    header_matching = [['dt', 'dt'], ['dt_iso', 'dt_iso'], ['timezone', 'timezone'], ['city_name', 'name'],
                       ['lat', 'coord_lat'],
                       ['lon', 'coord_lon'], ['temp', 'main_temp'], ['visibility', 'visibility'],
                       ['dew_point', 'dew_point'],
                       ['feels_like', 'feels_like'], ['temp_min', 'main_temp_min'], ['temp_max', 'main_temp_max'],
                       ['pressure', 'main_pressure'], ['sea_level', 'sea_level'], ['grnd_level', 'grnd_level'],
                       ['humidity', 'main_humidity'], ['wind_speed', 'wind_speed'], ['wind_deg', 'wind_deg'],
                       ['wind_gust', 'wind_gust'], ['rain_1h', 'rain_1h'], ['rain_3h', 'rain_3h'],
                       ['snow_1h', 'snow_1h'],
                       ['snow_3h', 'snow_3h'], ['clouds_all', 'clouds_all'], ['weather_id', 'weather_0_id'],
                       ['weather_main', 'weather_0_main'], ['weather_description', 'weather_0_description'],
                       ['weather_icon', 'weather_0_icon'], ['base', 'base']]

    one_h_weather_data = {}
    for i in range(len(header_matching)):
        try:
            one_h_weather_data[header_matching[i][0]] = contents[header_matching[i][1]]
        except:
            one_h_weather_data[header_matching[i][0]] = np.nan
            print('warning: ', header_matching[i][0])
    return one_h_weather_data



def main():
    # open existing weather df from /home/chrei/code/quantifiedSelfData/2022/weather_api_append.csv
    weather_df = pd.read_csv('/home/chrei/code/quantifiedSelfData/2022/weather_api_append.csv')
    daily_summaries_file_dir = "/home/chrei/code/quantifiedSelfData/daily_summaries_test_weather_api.csv"

    key = '36d299b06fa44c769f9174159222906'
    key_open_weather = 'a05b3d565a7f3b4a9c4f08df4c6df9f0'
    long = "48.74309462845568"
    lat = "9.101391671042892"
    date = "2022-06-28"
    one_h_weather_data = get_one_h_weather_data(long, lat, key_open_weather)


    # add the hour weather dict to the weather df
    weather_df = weather_df.append(one_h_weather_data, ignore_index=True)

    # make sure it's the correct date
    if contents['forecast']['forecastday'][0]['date'] != date:
        ValueError("Date does not match")

    # open daily_summaries_file_dir with pandas and set column Date as index
    daily_summaries_df = pd.read_csv(daily_summaries_file_dir, index_col=0)

    # check weather date exists in daily_summaries_df as index, if not create it
    if date not in daily_summaries_df.index:
        # if not, add it
        daily_summaries_df.loc[date] = np.nan

    # write max temp to daily_summaries_file_dir at the right date index.
    daily_summaries_df.loc[date, 'TempOutMax()'] = contents['forecast']['forecastday'][0]['day']['maxtemp_c']

    print()
