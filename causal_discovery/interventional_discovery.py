import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr

from config import verbosity_thesis, tau_max, target_label, interventional_discovery_on
from data_generation import labels_to_ints


def interventional_pass_filter(ts, was_intervened, tau):
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


def get_interventional_data_per_var(df, was_intervened, tau):
    # drop observational samples
    df, was_intervened = interventional_pass_filter(df, was_intervened, tau)

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


def add_median_non_interventional_data(cause_and_effect_tau_shifted, df, cause, effect, n_ini_obs):
    """add median non-interventional data to interventional data which will allow to see if there is a difference"""

    # get median of un-intervened data of the first n_ini_obs samples
    median_unintervened_data = df.iloc[:n_ini_obs].median()

    interventional_samples = len(cause_and_effect_tau_shifted)
    median_non_intervened_cause_effect = np.array(
        [[median_unintervened_data[cause], median_unintervened_data[effect]] for i in range(interventional_samples)])
    # v-stack cause_and_effect_tau_shifted with median_non_intervened_cause_effect then to df with headers
    cause_and_effect_tau_shifted = pd.DataFrame(
        np.vstack((cause_and_effect_tau_shifted, median_non_intervened_cause_effect)), columns=['cause', 'effect'])
    return cause_and_effect_tau_shifted


def get_probable_parents(effect, pag_edgemarks, measured_labels):
    """
    get probable parents of effect variable
    output: list of ['var', 'tau']
    probable parent has edgemark in ['-->', 'o->', 'x->']

    """
    probable_parents = []
    effect_int = labels_to_ints(measured_labels, effect)
    # iterate over tau
    for tau in range(0, tau_max + 1):
        # search for causes is column of effect var
        effect_col = pag_edgemarks[:, effect_int, tau]
        for item_idx in range(len(effect_col)):
            item = effect_col[item_idx]
            if item in ['-->', 'o->', 'x->']:
                probable_parents.append([measured_labels[item_idx], str(tau)])
        # search for causes is row of effect var
        effect_row = pag_edgemarks[effect_int, :, tau]
        for item_idx in range(len(effect_row)):
            item = effect_row[item_idx]
            if item in ['<--', '<-o', '<-x']:
                probable_parents.append([measured_labels[item_idx], str(tau)])

    # remove duplicates
    if len(probable_parents) > 0:
        found_duplicate = True
        while found_duplicate:
            found_duplicate = False
            for parents_idx in range(len(probable_parents)):
                parents = probable_parents[parents_idx]
                # count how often parents is in probable_parents
                count = 0
                for parents_idx2 in range(len(probable_parents)):
                    if parents == probable_parents[parents_idx2]:
                        count += 1
                if count > 1:
                    found_duplicate = True
                    probable_parents.pop(parents_idx)
                    break
    return probable_parents


def remove_cause_tau_var(probable_parents, cause, tau):
    # in probable_parents remove item if item == [cause, tau]
    probable_parents = list(probable_parents)
    tau=str(tau)
    for item_idx in range(len(probable_parents)):
        item = probable_parents[item_idx]
        if item[0] == cause and item[1] == tau:
            # remove item from probable_parents
            probable_parents.pop(item_idx)
            break
    return probable_parents

def get_conditioning_df(probable_parents, df, measured_labels):
    """
    get probable_parents' data and shift by tau
    """
    if len(probable_parents) <1:
        # return empty df
        return pd.DataFrame()
    else:
        # get conditioning set
        conditioning_df = []
        column_names = []
        for probable_parent in probable_parents:
            to_add = df.loc[:, probable_parent[0]].copy().shift(periods=int(probable_parent[1]))
            conditioning_df.append(to_add)

            # column names in format 'cause_tau'
            column_names.append(probable_parent[0] + '_' + probable_parent[1])

        # convert to dataframe
        conditioning_df = pd.DataFrame(np.array(conditioning_df).T, columns=column_names)
        return conditioning_df


def align_cause_effect_due_to_lag(cause_and_effect, tau):
    # ini tau shifted var
    cause_and_effect_tau_shifted = cause_and_effect.copy()
    if tau == 0:
        return cause_and_effect_tau_shifted
    else:
        # shift cause down by tau, to emulate contemporaneous cause
        cause_and_effect_tau_shifted['cause'] = cause_and_effect_tau_shifted[
            'cause'].copy().shift(periods=tau)
        return cause_and_effect_tau_shifted


def remove_weaker_links_of_contempt_cycles(dependencies_from_interv_data):
    # get contemporaneous links
    contemporaneous_links = []
    for var in dependencies_from_interv_data:
        if var[2] == 0:
            contemporaneous_links.append(var)


    # for all cont links
    removed_link = True
    while removed_link:
        removed_link = False
        for contemporaneous_link in contemporaneous_links:
            # remove current link and check if there is a reverse link
            cont_links_wo_this_link = [link for link in contemporaneous_links if link != contemporaneous_link]
            for cont_link_wo_this_link in cont_links_wo_this_link:
                if cont_link_wo_this_link[0] == contemporaneous_link[1] and cont_link_wo_this_link[1] == contemporaneous_link[0]:
                    # remove link with higher p-value
                    if cont_link_wo_this_link[3] > contemporaneous_link[3]:
                        contemporaneous_links.remove(cont_link_wo_this_link)
                        dependencies_from_interv_data.remove(cont_link_wo_this_link)
                        cont_links_wo_this_link.remove(cont_link_wo_this_link)
                    else:
                        contemporaneous_links.remove(contemporaneous_link)
                        dependencies_from_interv_data.remove(contemporaneous_link)
                    removed_link = True
                    break
            if removed_link:
                break


    return dependencies_from_interv_data


def get_interv_tau_aligned_cause_eff_df(d, w, cause, eff, tau):
    """
    input:
    d: dataframe with cause and effect
    w: was intervened df
    cause: cause variable
    eff: effect variable
    tau: time delay
    """
    # ignore contemporaneous auto dependencies: return empty df if eff == cause and tau == 0
    if (eff == cause and tau == 0) or (w[cause]==False).all():
        return pd.DataFrame([], columns=[cause, eff+'_'+str(tau)])

    # get indices where w[cause] == True
    interv_cause_dates = d.loc[:, cause].copy().loc[w[cause]].index

    cause_eff_df = []
    for date in interv_cause_dates:
        # pass effect val is interv value. if w at column eff and index date + tau is true
        if date + tau > d.index[-1]:
            continue
        if w.loc[date + tau, eff]:
            continue
        # get cause and effect data for date + tau
        cause_eff_df.append([d.loc[date, cause].copy(),d.loc[date + tau, eff].copy()])

    # cause_eff_df to df with columns cause and effect
    cause_eff_df = pd.DataFrame(cause_eff_df, columns=[cause, eff+'_'+str(tau)])

    return cause_eff_df




def get_independencies_from_interv_data(df, was_intervened, pc_alpha):
    """
    orient links with interventional data.
    test conditional independence for each var to each other var for all taus.
    output: (effect, cause or intervened, tau, p-val)
    """
    if interventional_discovery_on == False:
        return [], []

    independencies_from_interv_data = []
    dependencies_from_interv_data = []
    # iterate over all taus
    for tau in range(tau_max + 1):

        # # get interventional data per variable
        # interventional_dict = get_interventional_data_per_var(df, was_intervened, tau)

        # ini dependencies list


        # iterate over causes/interventions
        for cause in df.columns:

            # # stop if less than 3 samples, as corr coeff is not defined
            # if len(interventional_dict[cause]) > 2:

                # get data where on one specific var was intervened on
                # df_with_intervention_on_one_cause = interventional_dict[cause]

                # get values of cause var
                # cause_values = df_with_intervention_on_one_cause[cause]

                # skip if cause var is const, as corr is not defined
                # if len(np.unique(cause_values)) > 1:

            # iterate over all other (potentially effect) variables
            for effect in df.columns:

                # probable_parents = get_probable_parents(effect, pag_edgemarks, measured_labels)

                # get values of effect var
                # effect_values = df_with_intervention_on_one_cause[effect]

                # cause and effect series as columns in df
                # cause_and_effect = pd.DataFrame(dict(cause=cause_values, effect=effect_values))

                # add median non-interventional data to interventional data which will allow to see if there is a difference
                # cause_and_effect_non_interv_added = add_median_non_interventional_data(cause_and_effect.copy(), df, # todo probably not correct
                #                                                                   cause, effect, n_ini_obs)
                interv_tau_aligned_cause_eff_df = get_interv_tau_aligned_cause_eff_df(df, was_intervened, cause, effect, tau)

                # ignore contemporaneous auto-dependencies and data needs to be at least 3 (due to corr) + tau (shift drop nan) long
                if len(interv_tau_aligned_cause_eff_df) <5:
                    continue

                # get conditioning variables
                # conditioning_vars = remove_cause_tau_var(probable_parents, cause, tau)

                # returns probable edgemarks of effect variable with format [probable parent, tau]
                # conditioning_df = get_conditioning_df(conditioning_vars, df_with_intervention_on_one_cause,
                #                                         measured_labels)

                # emulate contemporaneous by shifting cause down by tau
                # cause_and_effect_tau_shifted = align_cause_effect_due_to_lag(cause_and_effect, tau)

                # add conditioning_set to cause_and_effect_tau_shifted as columns
                # cause_and_effect_condition_tau_shifted = pd.concat(
                #     [cause_and_effect_tau_shifted.copy(), conditioning_df], axis=1)
                # cause_and_effect_condition_tau_shifted = cause_and_effect_condition_tau_shifted.dropna()

                # if len(cause_and_effect_condition_tau_shifted) > 2:

                    # get p_val
                    # ans = pg.partial_corr(data=cause_and_effect_condition_tau_shifted, x='cause', y='effect',
                    #                       covar=list(conditioning_df.columns)).round(3)
                    # p_val = ans['p-val'].values[0] # probability of independence
                    # r = ans['r'].values[0] # correlation coefficient
                    # statistical test
                r, p_val = pearsonr(interv_tau_aligned_cause_eff_df[cause],
                                                      interv_tau_aligned_cause_eff_df[effect+'_'+str(tau)])
                # if significantly independent:
                if p_val > 1-pc_alpha: #interv_alpha:

                    # save independency information # (effect == '4' and tau == 0) or (effect == '2' and tau == 1) or (effect == '0' and tau == 0) or (effect == '3' and tau == 1)
                    independencies_from_interv_data.append((effect, cause, tau, p_val.round(4)))
                    if verbosity_thesis > 2:
                        print("interv discovery:", cause,
                              " -X>", effect, "with lag=", tau, ", p-value=",
                              p_val)
                    elif verbosity_thesis > 0:
                        if effect == target_label:
                            print("interv discovery: ", cause,
                                  " -X> target with lag", tau, "\t, p-value=",
                                  p_val)
                # if significantly dependent:
                elif p_val < pc_alpha:#(1-interv_alpha)*3:
                    dependencies_from_interv_data.append((effect, cause, tau, p_val.round(4)))

    # if contemporaneus cycle in dependencies_from_interv_data, remove link with weaker p-value
    dependencies_from_interv_data = remove_weaker_links_of_contempt_cycles(dependencies_from_interv_data)

    # print
    for dependency in dependencies_from_interv_data:
        if verbosity_thesis > 2:
            print("interv discovery: intervened var ", dependency[1],
                  " -->", dependency[0], "with lag=", dependency[2], ", p-value=",
                  dependency[3])
        elif verbosity_thesis > 0:
            if dependency[1] == target_label:
                print("interv discovery: ", dependency[1],
                      " --> target with lag", dependency[2], "\t, p-value=",
                      dependency[3])
    return independencies_from_interv_data, dependencies_from_interv_data

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
# pag_edgemarks = np.array([[['', '-->'], ['-->', '-->'], ['', '-->'], ['', '<->'], ['', '']],
#                           [['<--', '-->'], ['', '-->'], ['-->', '-->'], ['<->', ''], ['-->', '']],
#                           [['', '-->'], ['<--', '-->'], ['', '-->'], ['', ''], ['<->', '-->']],
#                           [['', 'o->'], ['<->', ''], ['', '<->'], ['', '<->'], ['', '']],
#                           [['', '<->'], ['<--', '<->'], ['<->', '-->'], ['', ''], ['', '-->']]])
# measured_labels=['0', '2', '3', '4', '5']
# independencies_from_interv_data = get_independencies_from_interv_data(
#     df=ts,
#     was_intervened=was_intervened,
#     interv_alpha=0.1,
#     n_ini_obs=500,
#     pag_edgemarks=pag_edgemarks,
#     measured_labels=measured_labels)
# print()
