import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from time import time

from tqdm import tqdm

from config import verbosity_thesis, target_label, tau_max, \
    n_samples, low_percentile, high_percentile
from data_generation import data_generator
from intervention_proposal.target_eqs_from_pag import plot_graph, drop_redundant_information_due_to_symmetry, \
    get_ambiguous_graph_locations, create_all_graph_combinations


def lin_f(x):
    return x


def graph_to_scm(my_graph, val):
    """
    input: graph, val
    output: scm
    """
    scm = {}
    for effect in range(len(my_graph[0])):
        scm.update({effect: []})
        effect_list = []
        for cause in range(len(my_graph)):
            # scm.update({cause: []})
            for tau in range(len(my_graph[cause][effect])):
                if my_graph[cause][effect][tau] in ['', '<--']:
                    continue
                elif my_graph[cause][effect][tau] == '-->':
                    effect_list.append(((cause, -tau), val[cause][effect][tau],lin_f))
                else:
                    ValueError('graph[cause][effect][tau] not in ["", "-->", "<--"]')
        scm[effect] = effect_list
    return scm


def drop_edges_for_cycle_detection(my_graph):
    """ set lagged edgemarks to "".
    and set contemporaneous edgemarks to "" """
    my_graph_without_lagged_variables = my_graph.copy()
    for cause in range(len(my_graph_without_lagged_variables)):
        for effect in range(len(my_graph_without_lagged_variables[cause])):
            for tau in range(len(my_graph_without_lagged_variables[cause][effect])):
                if tau > 0:
                    my_graph_without_lagged_variables[cause][effect][tau] = ""
                if tau == 0:
                    if my_graph_without_lagged_variables[cause][effect][tau] == '<->':
                        my_graph_without_lagged_variables[cause][effect][tau] = ''
    return my_graph_without_lagged_variables


def get_optimistic_intervention_var_via_simulation(val, my_graph, var_names, ts_old, unintervenable_vars, random_seed):
    """
    compute target equations of all graph combinations
    input: val_min, graph, var_names (loads from file)
    output: target_equations_per_graph_dict
    """
    # measure how long get_optimistic_intervention_var_via_simulation takes
    start_time = time()
    if verbosity_thesis > 0:
        print('get optimistic_intervention_var_via_simulation ...')

    # plot graph
    plot_graph(val, my_graph, var_names, 'current graph estimate')

    # drop redundant info in graph
    my_graph = drop_redundant_information_due_to_symmetry(my_graph)


    # find ambiguous link locations
    ambiguous_locations = get_ambiguous_graph_locations(my_graph)

    # create a list of all unique graph combinations
    graph_combinations = create_all_graph_combinations(my_graph, ambiguous_locations)
    print('len(graph_combinations): ', len(graph_combinations))

    n_half_samples = int(n_samples/2)

    largest_abs_coeff = 0
    largest_coeff = 0
    best_intervention_var_name = None
    most_optimistic_graph_idx = None


    for unique_graph_idx in tqdm(range(len(graph_combinations))):
        unique_graph = graph_combinations[unique_graph_idx]
        model = graph_to_scm(unique_graph, val)

        for intervention_var in var_names:
            # skip unintervenable intervention_vars like target label
            if intervention_var not in unintervenable_vars:
                samples = np.zeros(shape=(n_samples,len(var_names)))
                intervention_value_low = np.percentile(a=ts_old[intervention_var], q=low_percentile)
                intervention_value_high = np.percentile(a=ts_old[intervention_var], q=high_percentile)
                # intervene on intervention_var with low and high values
                simulated_data = data_generator(
                    scm=model,
                    intervention_variable=intervention_var,
                    intervention_value=intervention_value_low,
                    ts_old=ts_old,
                    random_seed=random_seed,
                    n_samples=n_half_samples,
                    labels=ts_old.columns
                )

                # if none then cyclic contemporaneous graph and skipp this graph
                if simulated_data is not None:
                    samples[0:n_half_samples] = simulated_data
                    samples[n_half_samples:n_samples] = data_generator(
                        scm=model,
                        intervention_variable=intervention_var,
                        intervention_value=intervention_value_high,
                        ts_old=ts_old,
                        random_seed=random_seed,
                        n_samples=n_half_samples,
                        labels=ts_old.columns
                    )

                    # for all tau
                    coeffs_across_taus = np.zeros(shape=(tau_max + 1))
                    for tau in range(tau_max + 1):

                        # intervention_var and target series as columns in df
                        samples = pd.DataFrame(samples, columns=var_names)
                        var_and_target = pd.DataFrame(
                            dict(intervention_var=samples[intervention_var], target=samples[target_label]))

                        # tau shift
                        if tau > 0:
                            var_and_target['target'] = var_and_target['target'].shift(periods=tau)
                            var_and_target = var_and_target.dropna()

                        # check var_and_target for NaNs and infs
                        if np.any(np.isnan(var_and_target)):
                            print('NaN in var_and_target')
                            print(var_and_target[var_and_target.isnull()])
                            raise ValueError('NaN in var_and_target')
                        if np.any(np.isinf(var_and_target)):
                            print('inf in var_and_target')
                            print(var_and_target[var_and_target.isinf()])
                            raise ValueError('inf in var_and_target')

                        # statistical test
                        r, probability_independent = pearsonr(var_and_target['intervention_var'],
                                                              var_and_target['target'])
                        coeffs_across_taus[tau] = r
                    mean_coeff_across_taus = np.mean(coeffs_across_taus)
                    if abs(mean_coeff_across_taus) > largest_abs_coeff:
                        largest_abs_coeff = abs(mean_coeff_across_taus)
                        largest_coeff = mean_coeff_across_taus
                        best_intervention_var_name = intervention_var
                        most_optimistic_graph_idx = unique_graph_idx



    # measure how long
    end_time = time()
    print('get optimistic_intervention_var_via_simulation took: ', end_time - start_time)
    if most_optimistic_graph_idx is None:
        mygraph_without_lagged = drop_edges_for_cycle_detection(my_graph)
        plot_graph(val, mygraph_without_lagged, var_names, 'contemp graph for cycle detection')
        print('WARNING: hack: no most optimistic graph found')

    return largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff, graph_combinations[most_optimistic_graph_idx]




#
# val_min, graph, var_names = load_results('chr')
# var_names = [str(x) for x in var_names]
#
# # save ts_old via pickle
# # with open(checkpoint_path+'ts_old.pickle', 'wb') as f:
# #     pickle.dump(ts_old, f)
#
# # load ts_old via pickle
# with open(checkpoint_path + 'ts_old.pickle', 'rb') as f:
#     ts_old = pickle.load(f)
# print('WARNING: loaded ts from pickle')
#
# largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff = get_optimistic_intervention_var_via_simulation(val_min, graph, var_names, ts_old, unintervenable_vars, random_seed)
#