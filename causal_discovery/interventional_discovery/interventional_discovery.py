import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr as pc
from scipy.stats import pearsonr


from causal_discovery.LPCMCI.lpcmci import LPCMCI
from causal_discovery.LPCMCI.observational_discovery import save_results
from config import verbosity, tau_max, pc_alpha, remove_link_threshold

"""
plain causal discovery
"""


def create_complete_graph(df):
    pass


def interventional_pass_filter(ts, was_intervened):
    """
    return only data with interventions
    """
    # iterate over all rows
    # create np.array of len 8


    interventional_data = np.empty((0,was_intervened.shape[1]))
    where_intervened = np.empty((0,was_intervened.shape[1]), dtype=bool)
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
        interventional_data_per_var[var] = np.empty((0,was_intervened.shape[1]))

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


def orient_link_with_interventional_data(interventional_dict):
    """
    orient links with interventional data.
    test each var to each other var for all taus.
    """
    independencies_in_interv_data= []
    for cause in interventional_dict:
        df_with_intervention_on_one_cause = interventional_dict[cause]
        cause_values = df_with_intervention_on_one_cause[cause]
        for effect in df_with_intervention_on_one_cause:
            effect_values = df_with_intervention_on_one_cause[effect]
            # make cause and effect series into df as columns
            cause_and_effect = pd.DataFrame(dict(cause=cause_values, effect=effect_values))


            cause_and_effect_tau_shifted = cause_and_effect.copy()
            for tau in range(tau_max+1):
                # if tau ==1:
                #     continue
                if (cause != effect) or (tau != 0): # ignore contemporaneous auto-dependencies

                    # data needs to be at least 3 (due to corr) + tau (shift drop nan) long
                    if len(cause_and_effect) > 2+tau:

                        # tau shift
                        if tau >0:
                            cause_and_effect_tau_shifted['effect'] = cause_and_effect['effect'].shift(periods=tau)
                            cause_and_effect_tau_shifted = cause_and_effect_tau_shifted.dropna()

                        # statistical test
                        r, probability_independent = pearsonr(df_with_intervention_on_one_cause[cause], df_with_intervention_on_one_cause[effect])

                        # if independency probability is above pc_alpha:
                        if probability_independent > pc_alpha:
                            # save independency information
                            independencies_in_interv_data.append((cause, effect, tau))
    return independencies_in_interv_data







def interv_discovery(df, was_intervened):


    # get interventional data per variable
    interventional_dict = get_interventional_data_per_var(df, was_intervened)





    # conduct independence tests
    orient_link_with_interventional_data(interventional_dict)

    # # standardize data
    # df -= df.mean(axis=0)
    # df /= df.std(axis=0)

    var_names = df.columns
    dataframe = pp.DataFrame(df.values, datatime=np.arange(len(df)),
                             var_names=var_names)

    lpcmci = LPCMCI(
        dataframe=dataframe,
        cond_ind_test=pc.ParCorr(
            significance='analytic',
            recycle_residuals=True))

    lpcmci.run_lpcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_p_non_ancestral=1,  # todo 3
        n_preliminary_iterations=1,  # todo 4
        prelim_only=False,
        verbosity=verbosity)

    graph = lpcmci.graph
    val_min = lpcmci.val_min_matrix

    # remove links if are below threshold
    graph[abs(val_min) < remove_link_threshold] = ""

    # plot predicted PAG
    tp.plot_graph(
        val_matrix=val_min,
        link_matrix=graph,
        var_names=var_names,
        link_colorbar_label='cross-MCI',
        node_colorbar_label='auto-MCI',
        figsize=(10, 6),
    )
    plt.show()

    # save results
    save_results(val_min, graph, var_names, 'simulated')
    return val_min, graph


# load ts dataframe from file
import os

filename = os.path.abspath("./../LPCMCI/tmp_test.dat")
ts = pd.read_csv(filename)

# get last row of ts and append to ts, and continue with index
last_row = ts.iloc[-1]
# ts = ts.append(last_row)
ts = ts.append(ts.iloc[-1], ignore_index=True)
ts.iloc[-2,0] = 9
ts.iloc[-3,0] = 8
ts.iloc[-1,5] = 9
ts.iloc[-4,0] = 9
ts.iloc[-5,0] = 8
ts.iloc[-6,0] = 9

## load was_intervened dataframe from file
filename = os.path.abspath("./../LPCMCI/tmp_was_intervened.dat")
was_intervened = pd.read_csv(filename)
was_intervened.iloc[-2,0] = True
was_intervened.iloc[-3,0] = True
was_intervened.iloc[-1,5] = True
was_intervened.iloc[-4,0] = True
was_intervened.iloc[-5,0] = True
was_intervened.iloc[-6,0] = True

pag_effect_sizes, pag_edgemarks = interv_discovery(
    df=ts,
    was_intervened=was_intervened)

print()
