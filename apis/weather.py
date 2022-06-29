import json
import urllib.request

import numpy as np
import pandas as pd

my_weather_headers = ['Daytime', 'TempOutMean()', 'TempOutMin()', 'TempOutMax()', 'TempOutDelta()', 'TempOutFeelMean()',
                      'TempFeelOutMin()', 'TempFeelOutMax()', 'PressOutMean()', 'PressOutMin()', 'PressOutMax()',
                      'PressOutDelta()', 'HumidOutMean()', 'HumidOutMin()', 'HumidOutMax()', 'WindOutMean()',
                      'WindOutMin()', 'WindOutMax()', 'CloudOutMean()', 'CloudOutMin()', 'CloudOutMax()', 'RainSnow'
                      ]

api_weather_headers = ['maxtemp_c', 'mintemp_c', 'avgtemp_c', 'maxwind_kph', 'totalprecip_mm', 'avgvis_km', 'avghumidity']

key = '36d299b06fa44c769f9174159222906'
long = "48.74309462845568"
lat = "9.101391671042892"
date = "2022-06-28"

daily_summaries_file_dir = "/home/chrei/code/quantifiedSelfData/daily_summaries_test_weather_api.csv"

contents = urllib.request.urlopen(
    "https://api.weatherapi.com/v1/history.json?key=" + key + "&q=" + long + "," + lat + "&dt=" + date).read()

# contents as json
contents = json.loads(contents)

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
