import os

import pandas as pd
from tqdm import tqdm

from helper import histograms

"""
go through csv files
create aggregation df for each file
select which aggregation is needed: max, min, mean, sum
append to data frame
write dataframe
"""
path_to_csv_files = '/home/chrei/code/quantifiedSelfData/2022/takeout-20220502T163423Z-001/Takeout/Fit/Daily activity metrics/'
outputname = './google_output_2022_05_12_percentile.csv'
verbose = False
print('running ...')


def csv_2_df(csv_root, csv_file, total_agg):
    """
    as name says
    """
    feature = os.path.splitext(csv_file)[0]
    if verbose:
        print('feature:', feature)
        print(csv_root, csv_file)
    google_csv_as_df = pd.read_csv(os.path.join(csv_root, csv_file))  # read csv to df
    if verbose:
        print('read df:', google_csv_as_df)
    # total_agg = total_agg.append(google_csv_as_df)



    # pd.DataFrame.quantile(0.5)
    # test = google_csv_as_df.agg(['median'], axis=0)
    aggregated = google_csv_as_df.agg(
        ['sum', 'min', 'max', 'mean'], axis=0)
    if verbose:
        print('column names (aggregated.columns):', aggregated.columns)

    # create dictionary
    daily_aggregation = {'Date': [feature]}

    if verbose:
        print('range(len(aggregated.columns)):', range(len(aggregated.columns)))
    i = -1
    for attribute_name in aggregated.columns:
        i += 1

        if aggregated.columns[i] == 'Start time':
            if verbose:
                print('skip Start time:')
        elif aggregated.columns[i] == 'End time':
            if verbose:
                print('skip End time:')
        elif aggregated.columns[i] == 'Move Minutes count':
            daily_aggregation[attribute_name] = [
                aggregated['Move Minutes count']['sum']]
        elif aggregated.columns[i] == 'Calories (kcal)':
            daily_aggregation[attribute_name] = [
                aggregated['Calories (kcal)']['sum']]
        elif aggregated.columns[i] == 'Distance (m)':
            daily_aggregation[attribute_name] = [
                aggregated['Distance (m)']['sum']]
        elif aggregated.columns[i] == 'Heart Points':
            daily_aggregation[attribute_name] = [
                aggregated['Heart Points']['sum']]
        elif aggregated.columns[i] == 'Heart Minutes':
            daily_aggregation[attribute_name] = [
                aggregated['Heart Minutes']['sum']]
        elif aggregated.columns[i] == 'Average heart rate (bpm)':
            daily_aggregation[attribute_name] = [
                aggregated['Average heart rate (bpm)']['mean']]
        elif aggregated.columns[i] == 'Max heart rate (bpm)':
            daily_aggregation[attribute_name] = [
                aggregated['Max heart rate (bpm)']['max']]
        elif aggregated.columns[i] == 'Min heart rate (bpm)':
            daily_aggregation[attribute_name] = [
                aggregated['Min heart rate (bpm)']['min']]
        elif aggregated.columns[i] == 'Median heart rate (bpm)':
            daily_aggregation[attribute_name] = [
                aggregated['Median heart rate (bpm)']['median']]
        elif aggregated.columns[i] == 'Low latitude (deg)':
            daily_aggregation[attribute_name] = [
                aggregated['Low latitude (deg)']['min']]
        elif aggregated.columns[i] == 'Low longitude (deg)':
            daily_aggregation[attribute_name] = [
                aggregated['Low longitude (deg)']['min']]
        elif aggregated.columns[i] == 'High latitude (deg)':
            daily_aggregation[attribute_name] = [
                aggregated['High latitude (deg)']['max']]
        elif aggregated.columns[i] == 'High longitude (deg)':
            daily_aggregation[attribute_name] = [
                aggregated['High longitude (deg)']['max']]
        elif aggregated.columns[i] == 'Average speed (m/s)':
            daily_aggregation[attribute_name] = [
                aggregated['Average speed (m/s)']['mean']]
        elif aggregated.columns[i] == 'Max speed (m/s)':
            daily_aggregation[attribute_name] = [
                aggregated['Max speed (m/s)']['max']]
        elif aggregated.columns[i] == 'Min speed (m/s)':
            daily_aggregation[attribute_name] = [
                aggregated['Min speed (m/s)']['min']]
        elif aggregated.columns[i] == 'Step count':
            daily_aggregation[attribute_name] = [
                aggregated['Step count']['sum']]
        elif aggregated.columns[i] == 'Average weight (kg)':
            daily_aggregation[attribute_name] = [
                aggregated['Average weight (kg)']['mean']]
        elif aggregated.columns[i] == 'Max weight (kg)':
            daily_aggregation[attribute_name] = [
                aggregated['Max weight (kg)']['max']]
        elif aggregated.columns[i] == 'Min weight (kg)':
            daily_aggregation[attribute_name] = [
                aggregated['Min weight (kg)']['min']]
        elif aggregated.columns[i] == 'Other duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Other duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Meditating duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Meditating duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Hiking duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Hiking duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Treadmill running duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Treadmill running duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Biking duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Biking duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Weight lifting duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Weight lifting duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Inactive duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Inactive duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Walking duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Walking duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Running duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Running duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Jogging duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Jogging duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Yoga duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Yoga duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Rowing machine duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Strength training duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Mountain biking duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'CrossFit duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Elliptical duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Roller skiing duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Stair climbing duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Stair climbing machine duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Swimming duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated[attribute_name]['sum']]
        elif aggregated.columns[i] == 'Sleep duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Sleep duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Light sleeping duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Light sleeping duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Deep sleeping duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Deep sleeping duration (ms)']['sum']]
        elif aggregated.columns[i] == 'REM sleeping duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['REM sleeping duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Awake mid-sleeping duration (ms)':
            daily_aggregation[attribute_name] = [
                aggregated['Awake mid-sleeping duration (ms)']['sum']]
        elif aggregated.columns[i] == 'Average systolic blood pressure (mmHg)':
            daily_aggregation['Average systolic blood pressure (mmHg)'] = [
                aggregated['Average systolic blood pressure (mmHg)']['mean']]
        elif aggregated.columns[i] == 'Body position':
            daily_aggregation[attribute_name] = [
                aggregated['Body position']['mean']]
        elif aggregated.columns[i] == 'Max systolic blood pressure (mmHg)':
            daily_aggregation[attribute_name] = [
                aggregated['Max systolic blood pressure (mmHg)']['max']]
        elif aggregated.columns[i] == 'Min systolic blood pressure (mmHg)':
            daily_aggregation[attribute_name] = [
                aggregated['Min systolic blood pressure (mmHg)']['min']]
        elif aggregated.columns[i] == 'Average diastolic blood pressure (mmHg)':
            daily_aggregation[
                attribute_name] = [aggregated['Average diastolic blood pressure (mmHg)']['mean']]
        elif aggregated.columns[i] == 'Max diastolic blood pressure (mmHg)':
            daily_aggregation[attribute_name] = [
                aggregated['Max diastolic blood pressure (mmHg)']['max']]
        elif aggregated.columns[i] == 'Min diastolic blood pressure (mmHg)':
            daily_aggregation[attribute_name] = [
                aggregated['Min diastolic blood pressure (mmHg)']['min']]
        elif aggregated.columns[i] == 'Average systolic blood pressure (mmHg)':
            daily_aggregation[
                attribute_name] = [aggregated['Average systolic blood pressure (mmHg)']['mean']]
        elif aggregated.columns[i] == 'Blood pressure measurement location':
            daily_aggregation[attribute_name] = [
                aggregated['Blood pressure measurement location']['mean']]
        else:
            print('!!! UNKNOWN LABEL !!! \n', aggregated.columns[i])
    if verbose:
        print('daily_aggregation:', daily_aggregation)

    daily_aggregation_df = pd.DataFrame(daily_aggregation)
    return daily_aggregation_df, total_agg


# create df from csv files
if verbose:
    print('path', path_to_csv_files)

columns = ['Date', 'Move Minutes count', 'Average systolic blood pressure (mmHg)', 'Max systolic blood pressure (mmHg)',
           'Min systolic blood pressure (mmHg)', 'Average diastolic blood pressure (mmHg)',
           'Max diastolic blood pressure (mmHg)', 'Min diastolic blood pressure (mmHg)', 'Body position',
           'Blood pressure measurement location', 'Calories (kcal)', 'Distance (m)', 'Heart Points', 'Heart Minutes',
           'Average heart rate (bpm)', 'Max heart rate (bpm)', 'Min heart rate (bpm)', 'Low latitude (deg)',
           'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)', 'Average speed (m/s)',
           'Max speed (m/s)', 'Min speed (m/s)', 'Step count', 'Average weight (kg)', 'Max weight (kg)',
           'Min weight (kg)', 'Inactive duration (ms)', 'Walking duration (ms)', 'Running duration (ms)',
           'Light sleeping duration (ms)', 'Deep sleeping duration (ms)', 'REM sleeping duration (ms)',
           'Awake mid-sleeping duration (ms)', 'Jogging duration (ms)', 'Sleep duration (ms)', 'Yoga duration (ms)',
           'Other duration (ms)', 'Biking duration (ms)', 'Treadmill running duration (ms)',
           'Weight lifting duration (ms)', 'Meditating duration (ms)', 'Rowing machine duration (ms)',
           'Stair climbing duration (ms)', 'Strength training duration (ms)', 'CrossFit duration (ms)',
           'Hiking duration (ms)', 'Mountain biking duration (ms)', 'Elliptical duration (ms)',
           'Swimming duration (ms)', 'Stair climbing machine duration (ms)']
total_agg = pd.DataFrame(columns=columns)

for root, dirs, files in os.walk(path_to_csv_files):
    print(len(files), 'files found')
    print('aggregating...might take a while...')

    i = 0
    for file in tqdm(files):  # go through all csv files
        if verbose:
            print('Filename=', file)
            print('root:', root)
            print('dirs:', dirs)
            print('files:', files)
            print('csv_2_df(root, file):', csv_2_df(root, file))
        if i == 0:
            df_0, total_agg = csv_2_df(root, file, total_agg)
        elif i == 1:
            df_1, total_agg = csv_2_df(root, file, total_agg)
            df = df_0.append(df_1)
        else:
            df_1, total_agg = csv_2_df(root, file, total_agg)
            df = df.append(df_1)
            if verbose:
                print(df)
                print('/n')
        i += 1

# histograms(total_agg.drop(['Start time', 'End time','Date','Jogging duration (ms)'], axis=1).rename(
#     columns={"Average speed (m/s)": "Average_speed", "Max speed (m/s)": "Max_speed",
#              "Min speed (m/s)": "Min_speed"}),
#     '/home/chrei/PycharmProjects/correlate/plots/raw_distributions/google')

# sort by Date
print('sorting...')
df = df.sort_values(by=['Date'])

# print file
print('writing...')
df.to_csv(outputname)
print('done! :)')

if verbose:
    print(df)
