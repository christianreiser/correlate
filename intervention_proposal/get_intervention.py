import itertools
import pickle

from matplotlib import pyplot as plt
from tigramite import plotting as tp
from tqdm import tqdm
from scipy.stats import norm

from config import checkpoint_path, private_folder_path


def load_results(name_extension):
    val_min = np.load(str(private_folder_path) + 'val_min_' + str(name_extension) + '.npy', allow_pickle=True)
    graph = np.load(str(private_folder_path) + 'graph_' + str(name_extension) + '.npy', allow_pickle=True)
    var_names = np.load(str(private_folder_path) + 'var_names_' + str(name_extension) + '.npy', allow_pickle=True)
    print('Attention: val_min, graph, var_names loaded from file')
    return val_min, graph, var_names


def load_eq():
    # load target_ans_per_graph_dict and graph_combinations from file via pickle
    with open(checkpoint_path + 'target_eq_simulated.pkl', 'rb') as f:
        target_eq = pickle.load(f)
    with open(checkpoint_path + 'graph_combinations_simulated.pkl',
              'rb') as f:
        graph_combinations = pickle.load(f)
    print("attention: target_eq and graph_combinations loaded from file")
    return target_eq, graph_combinations


from time import time

import numpy as np
import pandas as pd

from causal_discovery.LPCMCI.generate_data_mod import Graph
from config import verbosity_thesis, target_label, tau_max, percentile, n_samples_simulation
from data_generation import data_generator


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


def make_redundant_information_with_symmetry(graph, val):
    """
    make redundant link information of a graph with diagonal symmetry in matrix representation.
    e.g. A-->B = B<--A
    """
    # only lag zero
    tau = 0
    # if arrow is forward pointing insert symmetric backward arrow
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if i != j:
                # only modify if empty
                if graph[j, i, tau] == '':

                    if graph[i, j, tau] != '':
                        pass

                        if graph[i, j, tau] == '-->':
                            graph[j, i, tau] = '<--'
                        elif graph[i, j, tau] == '<--':
                            graph[j, i, tau] = '-->'
                        elif graph[i, j, tau] == 'x-x':
                            graph[j, i, tau] = 'x-x'
                        elif graph[i, j, tau] == '<->':
                            graph[j, i, tau] = '<->'
                        elif graph[i, j, tau] == 'o->':
                            graph[j, i, tau] = '<-o'
                        elif graph[i, j, tau] == '<-o':
                            graph[j, i, tau] = 'o->'
                        else:  # if arrow is not forward pointing, error
                            ValueError('Error: graph[i, j, tau] is not an arrow')
                        val[j, i, tau] = val[i, j, tau]

    return graph, val


def plot_graph(val_min, pag, my_var_names, link_colorbar_label):
    graph_redun, val_redun = make_redundant_information_with_symmetry(pag.copy(), val_min.copy())
    tp.plot_graph(
        val_matrix=val_redun,
        link_matrix=graph_redun,
        var_names=my_var_names,
        link_colorbar_label=link_colorbar_label,
        # node_colorbar_label='auto-MCI',
        figsize=(10, 6),
    )
    plt.show()


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


def get_all_tau_external_independencies_wrt_target(external_independencies, var_names):
    """
    if a var is in external dependencies for all tau w.r.t. target_label, then add it to is unintervenable
    """
    # external_independencies to list without lag
    external_independencies_str = []
    for external_independency in external_independencies:
        external_independencies_str.append(list(external_independency[0:-1]))

    # external_independencies to string labels
    for external_independency_idx in range(len(external_independencies_str)):
        external_independency = external_independencies_str[external_independency_idx]
        for var_idx in range(len(external_independency)):
            var = external_independency[var_idx]
            external_independencies_str[external_independency_idx][var_idx] = var_names[var]

    #
    all_tau_external_independencies_wrt_target = []
    for external_independency in external_independencies_str:
        # count how often the external independency in external_independencies_str
        if external_independency[1] == target_label:
            if external_independencies_str.count(external_independency) > tau_max:
                all_tau_external_independencies_wrt_target.append(external_independency[0])

    # remove duplicates
    all_tau_external_independencies_wrt_target = list(set(all_tau_external_independencies_wrt_target))
    return all_tau_external_independencies_wrt_target


def drop_redundant_information_due_to_symmetry(graph):
    """
    sometimes there is a redundant diagonal symmetry in matrix representation.
    if so, dropping values above diagonal
    """
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # drop all values of upper right triangle
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if i < j:
                    edge = graph[i, j, tau]
                    correspoding_edge = graph[j, i, tau]
                    if edge == '' and correspoding_edge == '':
                        pass
                    elif edge == '-->' and correspoding_edge == '<--':
                        graph[i, j, tau] = ''
                    elif edge == '<--' and correspoding_edge == '-->':
                        graph[i, j, tau] = ''
                    elif edge == 'o->' and correspoding_edge == '<-o':
                        graph[i, j, tau] = ''
                    elif edge == '<-o' and correspoding_edge == 'o->':
                        graph[i, j, tau] = ''
                    elif edge == '<->' and correspoding_edge == '<->':
                        graph[i, j, tau] = ''
                    elif edge == 'x->' and correspoding_edge == '<-x':
                        graph[i, j, tau] = ''
                    elif edge == '<-x' and correspoding_edge == 'x->':
                        graph[i, j, tau] = ''
                    elif edge == 'o-o' and correspoding_edge == 'o-o':
                        graph[i, j, tau] = ''
                    elif edge == 'x-x' and correspoding_edge == 'x-x':
                        graph[i, j, tau] = ''
                    elif edge == 'x-o' and correspoding_edge == 'o-x':
                        graph[i, j, tau] = ''
                    elif edge == 'o-x' and correspoding_edge == 'x-o':
                        graph[i, j, tau] = ''
                    else:
                        pass
    return graph


def get_ambiguous_graph_locations(graph):
    """
    1. Locate ambiguous edgemarks of a graph by string their i,j,k matrix indices.
    2. store their ambiguous original link.
    3. store their new possible unambiguous links.
    return:
    - [i, j, k, [original_links (ambiguous)], [[new_links (unambiguous)]]]
    - e.g. [0, 1, 0, ['o-o'], [["-->", " <->", "<--"]]]
    """
    # ambigious edgemark list
    ambiguous_edgemark_list = [
        "o->",  # -->, <->
        "x->",  # -->, <->
        "<-o",  # <--, <->
        "<-x",  # <--, <->
        "o-o",  # -->, <->, <--
        "x-x",  # -->, <->, <--
        "x-o",  # -->, <->, <--
        "o-x"]  # -->, <->, <--

    new_links_combinations = [
        ['-->', '<->'],
        ['-->', '<->'],
        ['<--', '<->'],
        ['<--', '<->'],
        ['-->', '<->', '<--'],
        ['-->', '<->', '<--'],
        ['-->', '<->', '<--'],
        ['-->', '<->', '<--']]

    ambiguous_locations = []  # [i, j, k, original_link, new_links
    # loop through all chars in graph:
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            for k in range(graph.shape[2]):
                original_link = graph[i, j, k]
                # for each char in string check if it is ambiguous
                if original_link in ambiguous_edgemark_list:
                    # get index of ambiguous edgemark
                    index = ambiguous_edgemark_list.index(original_link)

                    # append ambiguous location
                    ambiguous_locations.append([i, j, k, original_link, new_links_combinations[index]])
    return ambiguous_locations


def get_number_of_graph_combinations(ambiguous_locations):
    """
    get number of graph combinations
    input: ambiguous_locations
    - [i, j, k, original_link, new_links]
    - e.g. [0, 1, 0, ['o-o'], [["-->", " <->", "<--"]]]
    """
    number_of_graph_combinations = 1
    for ambiguous_location in ambiguous_locations:
        number_of_graph_combinations = number_of_graph_combinations * len(ambiguous_location[4])
    return number_of_graph_combinations


def get_unambiguous_links(ambiguous_locations):
    """
    for each ambiguous location get the list of new possible links
    return: list of lists of new links
    input: ambiguous_locations
    """
    # of every list in ambiguous_locations, get 4th element (new_links) in a new list
    unambiguous_links = []  # [i, j, k, new_links]
    for ambiguous_location in ambiguous_locations:
        unambiguous_links.append([ambiguous_location[4]])
    return unambiguous_links


def get_links_permutations(corresponding_unambiguous_links_list):
    """
    input: corresponding_unambiguous_links_list. each item is a list of unambiguous links corresponding to an ambiguous
        link.
    output: permutations_of_uniquified_links contains all links Permutations of possible links.
    """
    corresponding_unambiguous_links_list = [item for sublist in corresponding_unambiguous_links_list for item in
                                            sublist]
    permutations_of_uniquified_links = list(
        itertools.product(*corresponding_unambiguous_links_list))  # https://stackoverflow.com/a/2853239/7762867

    # if not links_permutations:
    #     links_permutations = np.transpose(np.asarray(corresponding_unambiguous_links_list[0]))
    return permutations_of_uniquified_links


def make_links_point_forward(graph):
    graph_forward = np.copy(graph)
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # if value == '<--', then switch i and j and change value to '-->'
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i, j, tau] == '<--':
                    graph_forward[i, j, tau] = ''
                    if graph[j, i, tau] != '' and i != j:
                        raise ValueError('graph[j, i, tau] != ''')
                    graph_forward[j, i, tau] = '-->'
    return graph_forward


def load_all_graph_combinations_from_file(graph_combinations, number_of_graph_combinations):
    for graph_idx in range(number_of_graph_combinations):
        with open(checkpoint_path + '{}.pkl'.format(
                graph_idx), 'rb') as f:
            graph_combinations[graph_idx] = pickle.load(f)
    return graph_combinations


def write_one_graph_combination_to_file(ambiguous_locations, links_permutations, graph_combinations, graph_idx):
    for ambiguous_location_idx in range(len(ambiguous_locations)):
        ambiguous_location = ambiguous_locations[ambiguous_location_idx]

        # get original link
        original_link = ambiguous_location[3]

        # get new links
        # new_links = ambiguous_location[4]

        # get i, j, k location
        i = ambiguous_location[0]
        j = ambiguous_location[1]
        k = ambiguous_location[2]

        new_link = links_permutations[graph_idx][ambiguous_location_idx]

        # get old link string
        old_link = graph_combinations[graph_idx][i, j, k]

        if i == j and new_link == '<--':
            ValueError('i == j and new_link == <--')

        # replace graph_combinations[graph_idx][i, j, k] with new_link string
        graph_combinations[graph_idx][i, j, k] = old_link.replace(original_link, new_link)

    # make links point forward
    graph_combinations[graph_idx] = make_links_point_forward(graph_combinations[graph_idx])

    # save graph_combinations[graph_idx] and graph_idx to file with pickle
    with open(
            checkpoint_path + '{}.pkl'.format(
                graph_idx), 'wb') as f:
        pickle.dump(graph_combinations[graph_idx], f)


def create_all_graph_combinations(graph, ambiguous_locations):
    """
    input: ambiguous_locations
    - [i, j, k, original_link, new_links]
    - e.g. [0, 1, 0, ['o-o'], [["-->", " <->", "<--"]]]
    """

    # if not ambiguous_locations:
    if ambiguous_locations is None or len(ambiguous_locations) == 0:
        return [graph]
    # if ambiguous_locations:
    else:
        # create number_of_graph_combinations original graphs
        number_of_graph_combinations = get_number_of_graph_combinations(ambiguous_locations)

        # initialize graph_combinations
        graph_combinations = []
        for combi_idx in range(number_of_graph_combinations):
            graph_combinations.append(np.copy(graph))

        corresponding_unambiguous_links_list = get_unambiguous_links(ambiguous_locations)
        links_permutations = get_links_permutations(corresponding_unambiguous_links_list)

        for graph_idx in range(number_of_graph_combinations):
            write_one_graph_combination_to_file(ambiguous_locations, links_permutations, graph_combinations, graph_idx)

        graph_combinations = load_all_graph_combinations_from_file(graph_combinations, number_of_graph_combinations)
        return graph_combinations


def find_optimistic_intervention(my_graph, val, ts, unintervenable_vars, random_seed,
                                 old_intervention, label, external_independencies,
                                 ):
    """
    Optimal control to find the most optimistic intervention.
    """

    # save my_graph, val, var_names, ts, unintervenable_vars, random_seed, old_intervention, label, external_independencies to file via pickle
    # with open(checkpoint_path + '{}.pkl'.format(label), 'wb') as f:
    #     pickle.dump([my_graph, val, var_names, ts, unintervenable_vars, random_seed, old_intervention, label, external_independencies], f)

    # measure how long get it takes
    start_time = time()

    if verbosity_thesis > 2:
        print('get optimistic_intervention_var_via_simulation ...')

    # don't intervene on variables that where independent of target var in interventional data for all taus,
    # by add them to unintervenable_vars
    if external_independencies is not None and len(external_independencies) > 0:
        to_add = get_all_tau_external_independencies_wrt_target(external_independencies, ts.columns)
        if to_add is not None and len(to_add) > 0:
            unintervenable_vars = unintervenable_vars + to_add

    # plot graph
    # if verbosity_thesis > 1 and label != 'true_scm':
    #     plot_graph(val, my_graph, var_names, 'current graph estimate')

    # drop redundant info in graph
    my_graph = drop_redundant_information_due_to_symmetry(my_graph)

    # find ambiguous link locations
    ambiguous_locations = get_ambiguous_graph_locations(my_graph)

    # create a list of all unique graph combinations
    graph_combinations = create_all_graph_combinations(my_graph, ambiguous_locations)

    n_half_samples = int(n_samples_simulation / 2)

    # get intervention value from fitted gaussian percentile
    intervention_value_low = {}
    intervention_value_high = {}
    for var_name in ts.columns:
        # Fit a normal distribution to the data:
        mu, std = norm.fit(ts[var_name])
        # get 95th percentile from normal distribution
        intervention_value_low[var_name] = norm.ppf(1 - percentile / 100, loc=mu, scale=std)
        intervention_value_high[var_name] = norm.ppf(percentile / 100, loc=mu, scale=std)


    largest_abs_coeff = 0
    largest_coeff = 0
    best_intervention_var_name = None
    most_optimistic_graph_idx = None
    most_optimistic_graph = None

    for unique_graph_idx, unique_graph in enumerate(tqdm(graph_combinations, position=0, leave=True)):
        model = graph_to_scm(unique_graph, val)

        # todo handle if pag does not contain an ACYCLIC graph, right leads to intervention = none. but could be better, esp for product
        # """start handle if graph has cont cycle"""
        # # ensure no contemporaneous cycles
        # check_contemporaneous_cycle(val, unique_graph, var_names, 'cycle check')

        # for all measured vars except unintervenable intervention_vars
        for intervention_var in list(set(ts.columns) - set(unintervenable_vars)):



            # intervene on intervention_var with low and high values
            simulated_low_interv, health = data_generator(
                scm=model,
                intervention_variable=intervention_var,
                intervention_value=intervention_value_low[intervention_var],
                ts_old=ts,
                random_seed=random_seed,
                n_samples=n_half_samples,
                labels=ts.columns,
                noise_type='without'
            )

            # get runtime type of simulated_low_interv


            # skip: cyclic contemporaneous graph (none) and 'max_lag == 0'
            if simulated_low_interv is not None and not isinstance(simulated_low_interv, str) and simulated_low_interv._typ == 'dataframe':

                simulated_low_interv = pd.DataFrame(simulated_low_interv, columns=ts.columns)
                sum_target_low_interv = simulated_low_interv[target_label]

                # same for high intervention value
                simulated_high_interv, health = data_generator(
                    scm=model,
                    intervention_variable=intervention_var,
                    intervention_value=intervention_value_high[intervention_var],
                    ts_old=ts,
                    random_seed=random_seed,
                    n_samples=n_half_samples,
                    labels=ts.columns,
                    noise_type='without',
                )
                simulated_high_interv = pd.DataFrame(simulated_high_interv, columns=ts.columns)
                sum_target_high_interv = simulated_high_interv[target_label]

                # get relative difference between low and high intervention
                coeff = (sum_target_high_interv - sum_target_low_interv).mean()

                # get absolute difference between low and high intervention
                abs_coeff = np.abs(coeff)

                # if abs_coeff > largest_abs_coeff:
                if abs_coeff > largest_abs_coeff:
                    largest_abs_coeff = abs_coeff
                    best_intervention_var_name = intervention_var
                    most_optimistic_graph_idx = unique_graph_idx
                    most_optimistic_graph = unique_graph
                    if coeff > 0:
                        intervention_value = intervention_value_high[intervention_var]
                    else:
                        intervention_value = intervention_value_low[intervention_var]

    # measure how long
    end_time = time()

    if most_optimistic_graph_idx is None:
        mygraph_without_lagged = drop_edges_for_cycle_detection(my_graph)
        plot_graph(val, mygraph_without_lagged, ts.columns, 'contemp graph for cycle detection')
        print('WARNING: hack: no most optimistic graph found')

    # # if intervention was found
    if best_intervention_var_name is not None:

        # plot most optimistic graph
        if verbosity_thesis > 1 and label != 'true_scm':
            plot_graph(val, most_optimistic_graph, ts.columns, 'most optimistic')

    # if intervention was not found
    else:
        print('WARNING: no intervention found. probably cyclic graph, or found no effect on target, var')
        best_intervention_var_name = old_intervention[0]
        intervention_value = old_intervention[1]

    if verbosity_thesis > 1:
        print("intervention_variable", label, best_intervention_var_name, "interv_val_opti: ",
              intervention_value)

    return best_intervention_var_name, intervention_value

# val_min, graph, var_names = load_results('chr')
# var_names = [str(x) for x in var_names]
#
# # save ts via pickle
# # with open(checkpoint_path+'ts.pickle', 'wb') as f:
# #     pickle.dump(ts, f)
#
# # load ts via pickle
# with open(checkpoint_path + 'ts.pickle', 'rb') as f:
#     ts = pickle.load(f)
# print('WARNING: loaded ts from pickle')
#
# largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff = get_optimistic_intervention_var_via_simulation(
#     val_min, graph, var_names, ts, unintervenable_vars, random_seed, external_independencies=external_independencies)
#
# # load val_min, graph, var_names, label from file via pickle
# with open(checkpoint_path + 'tmp.pkl', 'rb') as f:
#     val_min, graph, var_names, label = pickle.load(f)
# check_contemporaneous_cycle(val_min, graph, var_names, 'cycle check')

#
#
# unintervenable_vars = ['0', '1', '6']
# var_names = ['0', '2', '3', '4', '5']
# external_independencies = [(1, 0, 0), (1, 0, 1), (1, 4, 0)]
# get_all_tau_external_independencies_wrt_target(external_independencies, var_names)


# ############
