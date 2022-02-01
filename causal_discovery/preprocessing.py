# Imports

## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
from datetime import timedelta

import numpy as np
import pandas as pd


def remove_nan_seq_from_top_and_bot(df):
    for column in df:
        # reset index
        df = df.set_index('Date')
        df = df.reset_index()

        # array with indices of NaN entries
        indices_of_nans = df.loc[pd.isna(df[column]), :].index.values

        # remove unbroken sequence of nans from beginning of list
        sequence_number = -1
        for i in indices_of_nans:
            sequence_number += 1
            if i == sequence_number:
                df = df.drop([i], axis=0)
            else:
                break

        # remove unbroken sequence of nans from end of list

        # reset index
        df = df.set_index('Date')
        df = df.reset_index()

        indices_of_nans = df.loc[pd.isna(df[column]), :].index.values
        indices_of_nans = np.flip(indices_of_nans)
        len_df = len(df)
        sequence_number = len_df
        for i in indices_of_nans:
            sequence_number -= 1
            if i == sequence_number:
                df = df.drop([i], axis=0)
            else:
                break

        # print remaining nans
        remaining_nans = df.loc[pd.isna(df[column]), :].index.values
        if len(remaining_nans) > 0:
            print('remaining Nans in ' + str(column) + ': ', remaining_nans)

    return df


def non_contemporary_time_series_generation(df1):
    """
    so far works only when var 1 happens before far 2. E.G. sleep, exercise

    Parameters
    ----------
    df1[Date, var happening first, var happening second]: dataframe with format: one row per day

    Returns
    -------
    dataframe with format: 2 rows per day
    """
    # insert blanc row after every row
    df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d %H:%M')

    # df1['Date'] = datetime.strptime(df1['Date'], '%Y-%m-%dT% H:%M:%S.%f')
    df1.index = range(1, 2 * len(df1) + 1, 2)
    df = df1.reindex(index=range(2 * len(df1)))

    # modify df:
    # 1. add morning date time
    # 2. write heart points
    # 3. write sleep eff
    # 4. write evening date time
    for i, row in df.iterrows():
        if i % 2 == 0:
            if i != 0:
                df.loc[i, df.columns[0]] = df.loc[i - 1, df.columns[0]] + timedelta(hours=7, minutes=0)
                df.loc[i, df.columns[2]] = df.loc[i - 1, df.columns[2]]
            if i < len(df):
                df.loc[i, df.columns[1]] = df.loc[i + 1, df.columns[1]]

        else:  # i % 2 == 1:
            # df.loc[i, 'SleepEfficiency'] = df.loc[i+1, 'SleepEfficiency']
            df.loc[i, df.columns[0]] = df.loc[i, df.columns[0]] + timedelta(hours=23, minutes=0)

        # df.loc[i, 'HeartPoints'] = 1.0#df.loc[i-1, 'HeartPoints']
        # df.loc[i, 'SleepEfficiency'] = 1.0#df.loc[i+1, 'SleepEfficiency']

    df = df.iloc[1:]  # drop first row as it's missing data
    return df
