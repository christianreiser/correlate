import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config import verbosity_thesis, tau_max, pc_alpha


def interventional_pass_filter(ts, was_intervened):
    """
    return only data with interventions
    """
    # iterate over all rows
    # create np.array of len 8

    interventional_data = np.empty((0, was_intervened.shape[1]))
    where_intervened = np.empty((0, was_intervened.shape[1]), dtype=bool)
    for row in range(len(was_intervened)):
        # drop row if all its values are False
        if True in was_intervened.iloc[row].array:
            # add was_intervened_new to where_intervened as new row
            where_intervened = np.vstack((where_intervened, was_intervened.iloc[row]))
            # add ts_new to interventional_data as new row
            interventional_data = np.vstack((interventional_data, ts.iloc[row]))

    # to df
    # where_intervened to dataframe with columns of was_intervened
    where_intervened = pd.DataFrame(where_intervened, columns=was_intervened.columns)
    # interventional_data to dataframe with columns of ts
    interventional_data = pd.DataFrame(interventional_data, columns=ts.columns)
    return interventional_data, where_intervened


def get_interventional_data_per_var(df, was_intervened):
    # drop observational samples
    df, was_intervened = interventional_pass_filter(df, was_intervened)

    # ini dict with 2d array for each variable
    interventional_data_per_var = {}
    for var in was_intervened.columns:
        interventional_data_per_var[var] = np.empty((0, was_intervened.shape[1]))

    # fill dict with data
    for row in range(len(was_intervened)):
        for var in was_intervened.columns:
            if was_intervened.iloc[row][var]:
                interventional_data_per_var[var] = np.vstack((interventional_data_per_var[var], df.iloc[row]))

    # make dict of arrays to dict of dataframes
    interventional_dict_of_dfs = {}
    for var in interventional_data_per_var:
        interventional_dict_of_dfs[var] = pd.DataFrame(interventional_data_per_var[var], columns=df.columns)

    return interventional_dict_of_dfs


def get_independencies_from_interv_data(df, was_intervened):
    """
    orient links with interventional data.
    test each var to each other var for all taus.
    output: (causeing intervened var, intependent var, tau)
    """

    print('get_independencies_from_interv_data ...')
    # get interventional data per variable
    interventional_dict = get_interventional_data_per_var(df, was_intervened)

    independencies_from_interv_data = []
    for cause in interventional_dict:
        if len(interventional_dict[cause]) > 2:
            df_with_intervention_on_one_cause = interventional_dict[cause]
            cause_values = df_with_intervention_on_one_cause[cause]
            for effect in df_with_intervention_on_one_cause:
                effect_values = df_with_intervention_on_one_cause[effect]
                # make cause and effect series into df as columns
                cause_and_effect = pd.DataFrame(dict(cause=cause_values, effect=effect_values))

                cause_and_effect_tau_shifted = cause_and_effect.copy()
                for tau in range(tau_max + 1):
                    # if tau ==1:
                    #     continue
                    if (cause != effect) or (tau != 0):  # ignore contemporaneous auto-dependencies

                        # data needs to be at least 3 (due to corr) + tau (shift drop nan) long
                        if len(cause_and_effect) > 2 + tau:

                            # tau shift
                            if tau > 0:
                                cause_and_effect_tau_shifted['effect'] = cause_and_effect['effect'].shift(periods=tau)
                                cause_and_effect_tau_shifted = cause_and_effect_tau_shifted.dropna()

                            # statistical test
                            r, probability_independent = pearsonr(df_with_intervention_on_one_cause[cause],
                                                                  df_with_intervention_on_one_cause[effect])

                            # if independency probability is above pc_alpha:
                            if probability_independent > pc_alpha:
                                # save independency information
                                independencies_from_interv_data.append((cause, effect, tau))
                                if verbosity_thesis > 0:
                                    print("\nindependency in interventional data: intervened var ", cause,
                                          " is independent of var", effect, "with lag=", tau, ", p-value=",
                                          probability_independent)
    return independencies_from_interv_data




#
# # load ts dataframe from file
# import os
#
# filename = os.path.abspath("LPCMCI/tmp_test.dat")
# ts = pd.read_csv(filename)
#
# # get last row of ts and append to ts, and continue with index
# last_row = ts.iloc[-1]
# # ts = ts.append(last_row)
# ts = ts.append(ts.iloc[-1], ignore_index=True)
# ts.iloc[-2, 0] = 9
# ts.iloc[-3, 0] = 8
# ts.iloc[-1, 5] = 9
# ts.iloc[-4, 0] = 9
# ts.iloc[-5, 0] = 8
# ts.iloc[-6, 0] = 9
#
# ## load was_intervened dataframe from file
# filename = os.path.abspath("LPCMCI/tmp_was_intervened.dat")
# was_intervened = pd.read_csv(filename)
# was_intervened.iloc[-2, 0] = True
# was_intervened.iloc[-3, 0] = True
# was_intervened.iloc[-1, 5] = True
# was_intervened.iloc[-4, 0] = True
# was_intervened.iloc[-5, 0] = True
# was_intervened.iloc[-6, 0] = True
#
# independencies_from_interv_data = get_independencies_from_interv_data(
#     df=ts,
#     was_intervened=was_intervened)
#
# print()
