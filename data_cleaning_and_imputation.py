from datetime import datetime

import pandas as pd


def data_cleaning_and_imputation(df, target_label):
    """
    drop nutrition
    interpolate weight and VO2Max
    drop first and last days, as data is missing TODO too much is lost
    drop days where target data is missing (i.e., mood)
    check for missing values
    """

    # drop gps location
    # df = df.drop(
    #     ['Low latitude (deg)', 'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)'], axis=1)

    # drop nutrition
    df = df.drop(
        ['sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in'], axis=1)

    # interpolate weight and vo2 max linearly
    df['weight'] = df['weight'].interpolate(method='linear')
    df['VO2Max'] = df['VO2Max'].interpolate(method='linear')

    # add yesterdays target
    target_yesterday = str(target_label) + '_yesterday'
    df[target_yesterday] = df[target_label]
    df[target_yesterday] = df[target_yesterday].shift(periods=1)

    # drop days without target entry or yesterdays target entry
    for day, _ in df.iterrows():
        # checks for NaN
        if df[target_label][day] != df[target_label][day] or df[target_yesterday][day] != df[target_yesterday][day]:
            df = df.drop(day)

    # check for missing values
    for attribute_name in df.columns:
        nan_data_true_false = pd.isnull(df[attribute_name])
        nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
        nan_dates = nan_data_true_false[nan_numeric_indices].index
        if len(nan_dates) > 0:
            print('WARNING: ', attribute_name, 'entry is missing on', nan_dates, '!')

    return df


def drop_sparse_days(df):
    # drop days where too much data is missing TODO too much is lost
    date_list = pd.date_range(datetime.strptime('2019-02-12', '%Y-%m-%d'), periods=615).tolist()
    date_list = date_list + pd.date_range(datetime.strptime('2021-06-16', '%Y-%m-%d'), periods=41).tolist()
    date_list = [day.strftime('%Y-%m-%d') for day in date_list]
    for date in date_list:
        try:
            df = df.drop(date, axis=0)
        except Exception as e:
            print(e)
    return df
