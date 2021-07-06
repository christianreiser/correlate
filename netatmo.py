import pandas as pd
from datetime import datetime
from statistics import mean
from math import isnan

outputname = '/home/chrei/code/quantifiedSelfData/netatmo.csv'
df = pd.read_csv('/home/chrei/code/quantifiedSelfData/Indoor_7_5_2021.csv')  # , index_col=0

currentDay = datetime.strptime(df['DateTime Berlin'][0], '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
lastDay = ''
t_min = []
t_max = []
t_mean = []
humidity_min = []
humidity_max = []
humidity_mean = []
co2_min = []
co2_max = []
co2_mean = []
noise_min = []
noise_max = []
noise_mean = []
pressure_min = []
pressure_max = []
pressure_mean = []

date_agg = []
t_min_agg = []
t_max_agg = []
t_mean_agg = []
humidity_min_agg = []
humidity_max_agg = []
humidity_mean_agg = []
co2_min_agg = []
co2_max_agg = []
co2_mean_agg = []
noise_min_agg = []
noise_max_agg = []
noise_mean_agg = []
pressure_min_agg = []
pressure_max_agg = []
pressure_mean_agg = []

cold_start = True
for _, row in df.iterrows():
    currentDay = datetime.strptime(row['DateTime Berlin'], '%Y/%m/%d %H:%M:%S').strftime('%Y/%m/%d')
    # ddatetime = row[1]
    if currentDay != lastDay:
        print(row['DateTime Berlin'])
        if not cold_start:
            """ save daily aggs"""
            date_agg.append(lastDay)
            t_min_agg.append(min(t_min))
            t_max_agg.append(max(t_max))
            t_mean_agg.append(round(mean(t_mean), 1))
            humidity_min_agg.append(min(humidity_min))
            humidity_max_agg.append(max(humidity_max))
            humidity_mean_agg.append(round(mean(humidity_mean), 1))
            co2_min_agg.append(min(co2_min))
            co2_max_agg.append(max(co2_max))
            co2_mean_agg.append(round(mean(co2_mean), 1))
            noise_min_agg.append(min(noise_min))
            noise_max_agg.append(max(noise_max))
            noise_mean_agg.append(round(mean(noise_mean), 1))
            pressure_min_agg.append(min(pressure_min))
            pressure_max_agg.append(max(pressure_max))
            pressure_mean_agg.append(round(mean(pressure_mean), 1))

        """reset current """
        t_min = []
        t_max = []
        t_mean = []
        humidity_min = []
        humidity_max = []
        humidity_mean = []
        co2_min = []
        co2_max = []
        co2_mean = []
        noise_min = []
        noise_max = []
        noise_mean = []
        pressure_min = []
        pressure_max = []
        pressure_mean = []
        cold_start = False

    """append current"""
    if not isnan(row['Temperature']):
        t_min.append(float(row['Temperature']))
    if not isnan(row['Temperature']):
        t_max.append(float(row['Temperature']))
    if not isnan(row['Temperature']):
        t_mean.append(float(row['Temperature']))
    if not isnan(row['Humidity']):
        humidity_min.append(float(row['Humidity']))
    if not isnan(row['Humidity']):
        humidity_max.append(float(row['Humidity']))
    if not isnan(row['Humidity']):
        humidity_mean.append(float(row['Humidity']))
    if not isnan(row['CO2']):
        co2_min.append(float(row['CO2']))
    if not isnan(row['CO2']):
        co2_max.append(float(row['CO2']))
    if not isnan(row['CO2']):
        co2_mean.append(float(row['CO2']))
    if not isnan(row['Noise']):
        noise_min.append(float(row['Noise']))
    if not isnan(row['Noise']):
        noise_max.append(float(row['Noise']))
    if not isnan(row['Noise']):
        noise_mean.append(float(row['Noise']))
    if not isnan(row['Pressure']):
        pressure_min.append(float(row['Pressure']))
    if not isnan(row['Pressure']):
        pressure_max.append(float(row['Pressure']))
    if not isnan(row['Pressure']):
        pressure_mean.append(float(row['Pressure']))

    lastDay = currentDay

""" save daily aggs"""
date_agg.append(lastDay)
t_min_agg.append(min(t_min))
t_max_agg.append(max(t_max))
t_mean_agg.append(round(mean(t_mean), 1))
humidity_min_agg.append(min(humidity_min))
humidity_max_agg.append(max(humidity_max))
humidity_mean_agg.append(round(mean(humidity_mean), 1))
co2_min_agg.append(min(co2_min))
co2_max_agg.append(max(co2_max))
co2_mean_agg.append(round(mean(co2_mean), 1))
noise_min_agg.append(min(noise_min))
noise_max_agg.append(max(noise_max))
noise_mean_agg.append(round(mean(noise_mean), 1))
pressure_min_agg.append(min(pressure_min))
pressure_max_agg.append(max(pressure_max))
pressure_mean_agg.append(round(mean(pressure_mean), 1))

df = pd.DataFrame(list(
    zip(date_agg, t_min_agg, t_max_agg, t_mean_agg, humidity_min_agg, humidity_max_agg, humidity_mean_agg,
        co2_min_agg, co2_max_agg, co2_mean_agg, noise_min_agg, noise_max_agg, noise_mean_agg, pressure_min_agg,
        pressure_max_agg, pressure_mean_agg)),
    columns=['date_indoor', 't_min_indoor', 't_max_indoor', 't_mean_indoor', 'humidity_min_indoor',
             'humidity_max_indoor', 'humidity_mean_indoor', 'co2_min_indoor', 'co2_max_indoor', 'co2_mean_indoor',
             'noise_min_indoor', 'noise_max_indoor', 'noise_mean_indoor', 'pressure_min_indoor', 'pressure_max_indoor',
             'pressure_mean_indoor'])

df.to_csv(outputname)
