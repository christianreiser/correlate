import json
import time
import urllib.request
import numpy as np
import pandas as pd

from keys import key_open_weather


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


def change_format(contents):
    # calc dt_iso from dt unix, UTC to iso date format: e.g., 1652310000 to 2022-05-11 23:00:00 +0000 UTC
    # change kelvin to celsius
    # contents['main_temp'] = contents['main_temp']
    # contents['main_temp_min'] = contents['main_temp_min']
    # contents['main_temp_max'] = contents['main_temp_max']
    # contents['main_feels_like'] = contents['main_feels_like']


    return contents


def get_current_weather_dict(long, lat, key_open_weather):
    # url
    open_weather_map_request_url = 'https://api.openweathermap.org/data/2.5/weather?lat=' + long + '&lon=' + lat + '&appid=' + key_open_weather
    # contents = urllib.request.urlopen(
    #     "https://api.weatherapi.com/v1/history.json?key=" + key + "&q=" + long + "," + lat + "&dt=" + date).read()

    # request, read and decode json
    current_weather = json.loads(urllib.request.urlopen(open_weather_map_request_url).read())
    # flatten nested json
    current_weather = flatten_data(current_weather)

    # format changes to match the weather df
    current_weather['dt_iso'] = pd.to_datetime(current_weather['dt'], unit='s').isoformat()

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

    current_weather_dict = {}
    for i in range(len(header_matching)):
        try:
            current_weather_dict[header_matching[i][0]] = current_weather[header_matching[i][1]]
        except:
            current_weather_dict[header_matching[i][0]] = np.nan
            print('warning: ', header_matching[i][0])
    return current_weather_dict


def append_current_weather(long,lat, hourly_weather_path):
    current_weather_dict = get_current_weather_dict(long, lat, key_open_weather)

    # add the hour weather dict to existing hourly weather csv: read, append, save
    weather_df = pd.read_csv(hourly_weather_path)
    weather_df = weather_df.append(current_weather_dict, ignore_index=True)
    weather_df.to_csv(hourly_weather_path, index=False)

    print()

def main():
    # measure time: start
    start_time = time.time()
    long = "48.74309462845568"
    lat = "9.101391671042892"
    hourly_weather_path = '/home/chrei/code/quantifiedSelfData/2022/weather_api_append.csv'

    # every hour get the weather data and append to the csv
    while True:
        append_current_weather(long,lat, hourly_weather_path)
        # stop time and print time
        print("--- %s seconds ---" % (time.time() - start_time))
        time.sleep(3600)

main()