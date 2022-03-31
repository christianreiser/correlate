from datetime import datetime
from math import isnan
from tqdm import tqdm

from config import private_folder_path
from helper import histograms

import numpy as np
import pandas as pd

outputname = str(private_folder_path)+'netatmo_daily_summaries.csv'
df = pd.read_csv('/home/chrei/PycharmProjects/correlate/0_data_raw/weather/Indoor_7_5_2021.csv')  # , index_col=0

# histograms
histograms(df.drop(['Timestamp','DateTime Berlin'], axis=1), '/home/chrei/PycharmProjects/correlate/plots/raw_distributions/')

currentDay = datetime.strptime(df['DateTime Berlin'][0], '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
lastDay = ''
t_min5 = []
t_max95 = []
t_median = []
t_median = []
humidity_min5 = []
humidity_max95 = []
humidity_median = []
co2_min5 = []
co2_max95 = []
co2_median = []
noise_min5 = []
noise_max95 = []
noise_median = []
pressure_min5 = []
pressure_max95 = []
pressure_median = []

date_agg = []
t_min5_agg = []
t_max95_agg = []
t_median_agg = []
t_median_agg = []
humidity_min5_agg = []
humidity_max95_agg = []
humidity_median_agg = []
co2_min5_agg = []
co2_max95_agg = []
co2_median_agg = []
noise_min5_agg = []
noise_max95_agg = []
noise_median_agg = []
pressure_min5_agg = []
pressure_max95_agg = []
pressure_median_agg = []

cold_start = True
for _, row in tqdm(df.iterrows()):
    currentDay = datetime.strptime(row['DateTime Berlin'], '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
    # ddatetime = row[1]
    if currentDay != lastDay:
        if not cold_start:
            """ save daily aggs"""
            date_agg.append(lastDay)
            t_min5_agg.append(round(np.percentile(t_min5, 5), 1))
            t_max95_agg.append(round(np.percentile(t_max95, 95), 1))
            t_median_agg.append(round(np.percentile(t_median, 50), 1))
            humidity_min5_agg.append(round(np.percentile(humidity_min5, 5), 1))
            humidity_max95_agg.append(round(np.percentile(humidity_max95, 95), 1))
            humidity_median_agg.append(round(np.percentile(humidity_median, 50), 1))
            co2_min5_agg.append(round(np.percentile(co2_min5, 5), 1))
            co2_max95_agg.append(round(np.percentile(co2_max95, 95), 1))
            co2_median_agg.append(round(np.percentile(co2_median, 50), 1))
            noise_min5_agg.append(round(np.percentile(noise_min5, 5), 1))
            noise_max95_agg.append(round(np.percentile(noise_max95, 95), 1))
            noise_median_agg.append(round(np.percentile(noise_median, 50), 1))
            pressure_min5_agg.append(round(np.percentile(pressure_min5, 5), 1))
            pressure_max95_agg.append(round(np.percentile(pressure_max95, 95), 1))
            pressure_median_agg.append(round(np.percentile(pressure_median, 50), 1))

        """reset current """
        t_min5 = []
        t_max95 = []
        t_median = []
        t_median = []
        humidity_min5 = []
        humidity_max95 = []
        humidity_median = []
        co2_min5 = []
        co2_max95 = []
        co2_median = []
        noise_min5 = []
        noise_max95 = []
        noise_median = []
        pressure_min5 = []
        pressure_max95 = []
        pressure_median = []
        cold_start = False

    """append current"""
    if not isnan(row['Temperature']):
        t_min5.append(float(row['Temperature']))
    if not isnan(row['Temperature']):
        t_max95.append(float(row['Temperature']))
    if not isnan(row['Temperature']):
        t_median.append(float(row['Temperature']))
    if not isnan(row['Temperature']):
        t_median.append(float(row['Temperature']))
    if not isnan(row['Humidity']):
        humidity_min5.append(float(row['Humidity']))
    if not isnan(row['Humidity']):
        humidity_max95.append(float(row['Humidity']))
    if not isnan(row['Humidity']):
        humidity_median.append(float(row['Humidity']))
    if not isnan(row['CO2']):
        co2_min5.append(float(row['CO2']))
    if not isnan(row['CO2']):
        co2_max95.append(float(row['CO2']))
    if not isnan(row['CO2']):
        co2_median.append(float(row['CO2']))
    if not isnan(row['Noise']):
        noise_min5.append(float(row['Noise']))
    if not isnan(row['Noise']):
        noise_max95.append(float(row['Noise']))
    if not isnan(row['Noise']):
        noise_median.append(float(row['Noise']))
    if not isnan(row['Pressure']):
        pressure_min5.append(float(row['Pressure']))
    if not isnan(row['Pressure']):
        pressure_max95.append(float(row['Pressure']))
    if not isnan(row['Pressure']):
        pressure_median.append(float(row['Pressure']))

    lastDay = currentDay

""" save daily aggs"""
date_agg.append(lastDay)
t_min5_agg.append(round(np.percentile(t_min5, 5), 1))
t_max95_agg.append(round(np.percentile(t_max95, 95), 1))
t_median_agg.append(round(np.percentile(t_median, 50), 1))
# t_median_agg.append(round(np.percentile(t_median), 1))
humidity_min5_agg.append(round(np.percentile(humidity_min5, 5), 1))
humidity_max95_agg.append(round(np.percentile(humidity_max95, 95), 1))
humidity_median_agg.append(round(np.percentile(humidity_median, 50), 1))
co2_min5_agg.append(round(np.percentile(co2_min5, 5), 1))
co2_max95_agg.append(round(np.percentile(co2_max95, 95), 1))
co2_median_agg.append(round(np.percentile(co2_median, 50), 1))
noise_min5_agg.append(round(np.percentile(noise_min5, 5), 1))
noise_max95_agg.append(round(np.percentile(noise_max95, 95), 1))
noise_median_agg.append(round(np.percentile(noise_median, 50), 1))
pressure_min5_agg.append(round(np.percentile(pressure_min5, 5), 1))
pressure_max95_agg.append(round(np.percentile(pressure_max95, 95), 1))
pressure_median_agg.append(round(np.percentile(pressure_median, 50), 1))

df = pd.DataFrame(list(
    zip(date_agg, t_min5_agg, t_max95_agg, t_median_agg, t_median_agg, humidity_min5_agg, humidity_max95_agg, humidity_median_agg,
        co2_min5_agg, co2_max95_agg, co2_median_agg, noise_min5_agg, noise_max95_agg, noise_median_agg, pressure_min5_agg,
        pressure_max95_agg, pressure_median_agg)),
    columns=['wInDate', 'wInTMin5', 'wInTMax95', 'wInTMedian', 'wInHumidityMin5',
             'wInHumidityMax95', 'wInHumidityMedian', 'wInCO2Min5', 'wInCO2Max95', 'wInCO2Median',
             'wInNoiseMin5', 'wInNoiseMax95', 'wInNoiseMedian', 'wInPressureMin5', 'wInPressureMax95',
             'wInPressureMedian'])

df.to_csv(outputname)
