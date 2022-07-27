import itertools

from matplotlib import pyplot as plt
from scipy.stats import norm
from tigramite import plotting as tp
from tqdm import tqdm

from config import private_folder_path


def load_results(name_extension):
    val_min = np.load(str(private_folder_path) + 'val_min_' + str(name_extension) + '.npy', allow_pickle=True)
    graph = np.load(str(private_folder_path) + 'graph_' + str(name_extension) + '.npy', allow_pickle=True)
    var_names = np.load(str(private_folder_path) + 'var_names_' + str(name_extension) + '.npy', allow_pickle=True)
    print('Attention: val_min, graph, var_names loaded from file')
    return val_min, graph, var_names


# def load_eq():
#     # load target_ans_per_graph_dict and graph_combinations from file via pickle
#     with open(checkpoint_path + 'target_eq_simulated.pkl', 'rb') as f:
#         target_eq = pickle.load(f)
#     with open(checkpoint_path + 'graph_combinations_simulated.pkl',
#               'rb') as f:
#         graph_combinations = pickle.load(f)
#     print("attention: target_eq and graph_combinations loaded from file")
#     return target_eq, graph_combinations


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
    # if arrow is forward pointing insert symmetric backward arrow
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if i != j:
                # only write in empty cells
                if graph[j, i, 0] == '':
                    pass
                    # pass if original cell also already empty
                    original_arrow = graph[i, j, 0]
                    if original_arrow == '':
                        pass
                    # insert symmetric arrow
                    elif original_arrow == '-->':
                        graph[j, i, 0] = '<--'
                    elif original_arrow == '<--':
                        graph[j, i, 0] = '-->'
                    elif original_arrow == 'x-x':
                        graph[j, i, 0] = 'x-x'
                    elif original_arrow == 'o-o':
                        graph[j, i, 0] = 'o-o'
                    elif original_arrow == '<->':
                        graph[j, i, 0] = '<->'
                    elif original_arrow == 'o->':
                        graph[j, i, 0] = '<-o'
                    elif original_arrow == '<-o':
                        graph[j, i, 0] = 'o->'
                    else:  # if arrow is not forward pointing, error
                        raise ValueError('Error: graph[i, j, tau] is not an arrow')

                    val[j, i, 0] = val[i, j, 0]

    return graph, val


def plot_graph(val_min, pag, my_var_names, link_colorbar_label, make_redundant):
    # filename = checkpoint_path + 'plot_error.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump([val_min, pag, my_var_names, link_colorbar_label], f)

    if make_redundant:
        graph_redun, val_redun = make_redundant_information_with_symmetry(pag.copy(), val_min.copy())
    else:
        graph_redun = pag.copy()
        val_redun = val_min.copy()
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
            plot_graph(val_min, cont_graph, var_names, 'contemp cycle detected', make_redundant=False)
        raise ValueError("Contemporaneous links must not contain cycle.")  # todo check if this always illegitimate


def get_all_tau_external_independencies_wrt_target(external_independencies, var_names):
    """
    if a var is in external dependencies for all tau w.r.t. target_label, then add it to is unintervenable
    """
    if external_independencies is None or len(external_independencies) > 0:
        return []

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





def get_one_graph_combination(ambiguous_locations, links_permutations, graph_combinations, graph_idx):
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

    return graph_combinations[graph_idx]


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

        # write graph combi
        for graph_idx in range(number_of_graph_combinations):
            graph_combinations[graph_idx] = get_one_graph_combination(ambiguous_locations, links_permutations, graph_combinations, graph_idx)

        return graph_combinations


def find_optimistic_intervention(my_graph, val, ts, unintervenable_vars, random_seed,
                                 label, external_independencies,
                                 ):
    """
    Optimal control to find the most optimistic intervention.
    """
    if verbosity_thesis > 2:
        print('get optimistic_intervention_var_via_simulation ...')

    # don't intervene on variables that where independent of target var in interventional data for all taus,
    # by add them to unintervenable_vars
    external_independencies_wrt_target = get_all_tau_external_independencies_wrt_target(external_independencies,
                                                                                        ts.columns)
    if len(external_independencies_wrt_target) > 0:
        unintervenable_vars = unintervenable_vars + external_independencies_wrt_target

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
    best_intervention_var_name = None
    most_optimistic_graph_idx = None
    most_optimistic_graph = None

    for unique_graph_idx, unique_graph in enumerate(tqdm(graph_combinations, position=0, leave=True, delay=10)):
        model = graph_to_scm(unique_graph, val)

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

            # skip: cyclic contemporaneous graph (none) and 'max_lag == 0'
            if health == 'good':

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
            elif health == 'max_lag == 0' and verbosity_thesis>2:
                print('skipped because max_lag == 0')
            elif health == 'cyclic contemporaneous scm':
                print('skipped because cyclic contemporaneous scm')


    if most_optimistic_graph_idx is None and verbosity_thesis >1:
        mygraph_without_lagged = drop_edges_for_cycle_detection(my_graph)
        plot_graph(val, mygraph_without_lagged, ts.columns, 'contemp graph for cycle detection', make_redundant=True)
        print('WARNING: hack: no most optimistic graph found')

    # # if intervention was found
    if best_intervention_var_name is not None:

        # plot most optimistic graph
        if verbosity_thesis > 1 and label != 'true_scm':
            plot_graph(val, most_optimistic_graph, ts.columns, 'most optimistic', make_redundant=True)

    # if intervention was not found
    else:
        print('WARNING: no intervention found. now ignoring scm directions')
        most_extreme_val, best_intervention_var_name = get_intervention_ignoring_directionalities(val.copy(),
                                                                                                  target_label,
                                                                                                  labels_as_str=ts.columns,
                                                                                                  external_independencies_wrt_target=external_independencies_wrt_target,
                                                                                                  ignore_external_independencies=False)
        if (most_extreme_val, best_intervention_var_name) == (None, None):
            most_extreme_val, best_intervention_var_name = get_intervention_ignoring_directionalities(val.copy(),
                                                       target_label,
                                                       labels_as_str=ts.columns,
                                                       external_independencies_wrt_target=external_independencies_wrt_target,
                                                       ignore_external_independencies=True)
        if most_extreme_val is None:
            best_intervention_var_name, intervention_value = None, None
        elif most_extreme_val > 0:
            intervention_value = intervention_value_high[best_intervention_var_name]
        elif most_extreme_val < 0:
            intervention_value = intervention_value_low[best_intervention_var_name]
        else:
            raise ValueError('most extreme value is 0')
    return best_intervention_var_name, intervention_value


def get_intervention_ignoring_directionalities(vals, var_name_as_str, labels_as_str,
                                               external_independencies_wrt_target, ignore_external_independencies):
    # fix int vs str format
    var = list(labels_as_str).index(var_name_as_str)
    if ignore_external_independencies:
        unintervenables_without_var = []
    else:
        unintervenables_without_var = [list(labels_as_str).index(var) for var in external_independencies_wrt_target]

    highest_abs_corr = 0
    most_extreme_val = 0
    most_extreme_var = []
    # set vals to zero if auto corr or non-target adjacent var or unintervenable var
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if i == j or (
                    i != var and j != var) or i in unintervenables_without_var or j in unintervenables_without_var:
                for tau in range(vals.shape[2]):
                    vals[i, j, tau] = 0
    # find max val in vals
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            for tau in range(vals.shape[2]):
                if np.abs(vals[i, j, tau]) > highest_abs_corr:
                    highest_abs_corr = np.abs(vals[i, j, tau])
                    most_extreme_val = vals[i, j, tau]
                    most_extreme_var = [i, j]

    # remove target from most extreme var
    for i in range(len(most_extreme_var)):
        if most_extreme_var[i] == var:
            most_extreme_var.pop(i)
            break
    if len(most_extreme_var) > 1:
        raise Exception('len(most_extreme_var) > 0')
    elif len(most_extreme_var) == 0:
        print('ignore_external_independencies is True')
        return None, None
    else: # len(most_extreme_var) == 1
        most_extreme_var = most_extreme_var[0]
    return most_extreme_val, labels_as_str[most_extreme_var]




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
