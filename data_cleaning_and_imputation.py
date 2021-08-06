from datetime import datetime

import pandas as pd


def data_cleaning_and_imputation(df, target_label):
    """
    interpolate weight and VO2Max
    add yesterdays and ereyesterdays mood as feature
    """

    # drop gps location
    # df = df.drop(
    #     ['Low latitude (deg)', 'Low longitude (deg)', 'High latitude (deg)', 'High longitude (deg)'], axis=1)

    # interpolate weight and vo2 max linearly
    try:
        df['weight'] = df['weight'].interpolate(method='linear')
    except:
        pass
    try:
        df['VO2Max'] = df['VO2Max'].interpolate(method='linear')
    except:
        pass

    # add yesterdays target
    target_yesterday = str(target_label) + '_yesterday'
    target_ereyesterday = str(target_label) + '_ereyesterday'
    df[target_yesterday] = df[target_label]
    df[target_ereyesterday] = df[target_label]
    df[target_yesterday] = df[target_yesterday].shift(periods=1)
    df[target_ereyesterday] = df[target_ereyesterday].shift(periods=2)

    # drop days without target entry or yesterdays target entry
    for day, _ in df.iterrows():
        # checks for NaN
        if df[target_label][day] != df[target_label][day] or df[target_yesterday][day] != df[target_yesterday][day] or \
                df[target_ereyesterday][day] != df[target_ereyesterday][day]:
            df = df.drop(day)

    return df


def drop_attributes_with_missing_values(df):
    # drop first and last days where data is not fully available
    date_list = pd.date_range(start=datetime.strptime('2021-06-15', '%Y-%m-%d'), periods=99).tolist()
    date_list = [day.strftime('%Y-%m-%d') for day in date_list]
    date_list.append(['2019-02-12', '2019-02-13'])
    for date in date_list:
        try:
            df = df.drop(date, axis=0)
        except Exception as e:
            # print(e)
            pass

    # drop attributes with missing values
    attribute_names = df.columns
    for attribute_name in attribute_names:
        nan_data_true_false = pd.isnull(df[attribute_name])
        nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
        nan_dates = nan_data_true_false[nan_numeric_indices].index
        if len(nan_dates) > 0:
            df = df.drop(attribute_name, axis=1)
    return df


def drop_days_with_missing_values(df):
    # drop nutrition
    df = df.drop(
        ['sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in'], axis=1)

    for attribute_name in df.columns:
        nan_data_true_false = pd.isnull(df[attribute_name])
        nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
        nan_dates = nan_data_true_false[nan_numeric_indices].index
        if len(nan_dates) > 0:
            df = df.drop(nan_dates, axis=0)
    return df


def missing_value_check(df):
    for attribute_name in df.columns:
        nan_data_true_false = pd.isnull(df[attribute_name])
        nan_numeric_indices = pd.isnull(df[attribute_name]).to_numpy().nonzero()[0]
        nan_dates = nan_data_true_false[nan_numeric_indices].index
        if len(nan_dates) > 0:
            print('WARNING: missing value ', nan_dates, attribute_name)
    return df


def drop_days_before__then_drop_col(df, last_day_to_drop):
    # drop nutrition
    df = df.drop(
        ['sodium', 'fat', 'carbohydrates', 'protein', 'fibre', 'kcal_in'], axis=1)

    # drop days where too much data is missing manually by picking dates
    date_list = pd.date_range(start=datetime.strptime('2019-02-11', '%Y-%m-%d'), end=last_day_to_drop).tolist()
    date_list = [day.strftime('%Y-%m-%d') for day in date_list]
    for date in date_list:
        try:
            df = df.drop(date, axis=0)
        except Exception as e:
            pass

    df = drop_attributes_with_missing_values(df)
    return df
