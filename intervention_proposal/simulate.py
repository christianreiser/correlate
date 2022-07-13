from time import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from causal_discovery.LPCMCI.generate_data_mod import Graph
from config import verbosity_thesis, target_label, tau_max, low_percentile, high_percentile, n_samples_simulation
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
                    effect_list.append(((cause, -tau), val[cause][effect][tau], lin_f))
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
    """ remove node if they don't have at least one incoming and outgoing edgemark, as they then cant be in the cycle"""
    # for cause in range(len(my_graph_without_lagged_variables)):
    #     if len(my_graph_without_lagged_variables[cause]) == 0:
    #         my_graph_without_lagged_variables.pop(cause)
    return my_graph_without_lagged_variables


def check_contemporaneous_cycle(val_min, graph, var_names, label):
    # # save val_min, graph, var_names, label to file via pickle
    # with open(checkpoint_path + 'tmp.pkl', 'wb') as f:
    #     pickle.dump([val_min, graph, var_names, label], f)

    links = graph_to_scm(graph, val_min)
    N = len(links.keys())

    # Check parameters
    max_lag = 0
    contemp_dag = Graph(N)
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]

            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)
    if contemp_dag.isCyclic() == 1:
        cycle_nodes = contemp_dag.get_cycle_nodes()
        cont_graph = drop_edges_for_cycle_detection(graph)
        if verbosity_thesis > 0:
            plot_graph(val_min, cont_graph, var_names, 'contemp cycle detected')
        raise ValueError("Contemporaneous links must not contain cycle.")  # todo check if this always illegitimate


def get_optimistic_intervention_var_via_simulation(val, my_graph, var_names, ts_old, unintervenable_vars, random_seed, label):
    """
    compute target equations of all graph combinations
    input: val_min, graph, var_names (loads from file)
    output: target_equations_per_graph_dict
    """
    # measure how long get it takes
    start_time = time()
    if verbosity_thesis > 2:
        print('get optimistic_intervention_var_via_simulation ...')

    # plot graph
    if verbosity_thesis > 1 and label != 'true_scm':
        plot_graph(val, my_graph, var_names, 'current graph estimate')

    # drop redundant info in graph
    my_graph = drop_redundant_information_due_to_symmetry(my_graph)

    # find ambiguous link locations
    ambiguous_locations = get_ambiguous_graph_locations(my_graph)

    # create a list of all unique graph combinations
    graph_combinations = create_all_graph_combinations(my_graph, ambiguous_locations)
    if verbosity_thesis > 3:
        print('len(graph_combinations): ', len(graph_combinations))

    n_half_samples = int(n_samples_simulation / 2)

    largest_abs_coeff = 0
    largest_coeff = 0
    best_intervention_var_name = None
    most_optimistic_graph_idx = None
    most_optimistic_graph = None

    for unique_graph_idx in range(len(graph_combinations)):
        unique_graph = graph_combinations[unique_graph_idx]
        model = graph_to_scm(unique_graph, val)

        # todo handle if pag does not contain an ACYCLIC graph, right leads to intervention = none. but could be better, esp for product
        # """start handle if graph has cont cycle"""
        # # ensure no contemporaneous cycles
        # check_contemporaneous_cycle(val, unique_graph, var_names, 'cycle check')

        for intervention_var in var_names:
            # skip unintervenable intervention_vars like target label
            if intervention_var not in unintervenable_vars:
                samples = np.zeros(shape=(n_samples_simulation, len(var_names)))
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
                    samples[n_half_samples:n_samples_simulation] = data_generator(
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
                        most_optimistic_graph = unique_graph

    # measure how long
    end_time = time()
    if verbosity_thesis > 2:
        print('get optimistic_intervention_var_via_simulation took: ', end_time - start_time)
    if most_optimistic_graph_idx is None:
        mygraph_without_lagged = drop_edges_for_cycle_detection(my_graph)
        plot_graph(val, mygraph_without_lagged, var_names, 'contemp graph for cycle detection')
        print('WARNING: hack: no most optimistic graph found')

    return largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff, most_optimistic_graph

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

# # load val_min, graph, var_names, label from file via pickle
# with open(checkpoint_path + 'tmp.pkl', 'rb') as f:
#     val_min, graph, var_names, label = pickle.load(f)
# check_contemporaneous_cycle(val_min, graph, var_names, 'cycle check')