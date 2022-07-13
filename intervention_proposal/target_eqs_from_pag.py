import itertools
import pickle
# from multiprocessing import Pool
#
import numpy as np
# import sympy as sp
from matplotlib import pyplot as plt
from tigramite import plotting as tp
# from tqdm import tqdm
#
# from config import target_label, private_folder_path, verbosity_thesis
#
# """
# This file contains the functions to compute the target equations from the PAG.
# The steps are:
# 1. load the PAG from a file
# 2. optional: plot graph
# 3. drop redundant information due to symmetry
# 4. get ambiguous locations
# 5. get number of graph combinations
# 6. get new links list
# 7. get links permutations
# 8. make links point forward
# 9. create all graph combinations
# 10. get direct influences of variables
# 11. add unknown noise to all equations
# 12. solve equations to get causes of target
# 13. module test
# """
#
#
# function that loads val_min, graph, and var_names from a file and allow_pickle=True
def load_results(name_extension):
    val_min = np.load(str(private_folder_path) + 'val_min_' + str(name_extension) + '.npy', allow_pickle=True)
    graph = np.load(str(private_folder_path) + 'graph_' + str(name_extension) + '.npy', allow_pickle=True)
    var_names = np.load(str(private_folder_path) + 'var_names_' + str(name_extension) + '.npy', allow_pickle=True)
    print('Attention: val_min, graph, var_names loaded from file')
    return val_min, graph, var_names


from config import checkpoint_path, private_folder_path, verbosity_thesis


def plot_graph(val_min, pag, var_names, label):
    graph = pag.copy()
    graph = make_redundant_information_with_symmetry(graph)
    tp.plot_graph(
        val_matrix=val_min,
        link_matrix=graph,
        var_names=var_names,
        link_colorbar_label=label,
        # node_colorbar_label='auto-MCI',
        figsize=(10, 6),
    )
    plt.show()

    # Plot time series graph
    # tp.plot_time_series_graph(
    #     figsize=(12, 8),
    #     val_matrix=val_min,
    #     link_matrix=graph,
    #     var_names=var_names,
    #     link_colorbar_label='MCI',
    # )
    # plt.show()


def drop_redundant_information_due_to_symmetry(graph):
    """
    drop redundant link information of a graph due to diagonal symmetry in matrix representation.
    meaning, dropping values above diagonal
    e.g. A-->B = B<--A
    """
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # drop all values of upper right triangle
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if i < j:
                    graph[i, j, tau] = ''
    return graph


def make_redundant_information_with_symmetry(graph):
    """
    make redundant link information of a graph with diagonal symmetry in matrix representation.
    e.g. A-->B = B<--A
    """
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # if arrow is forward pointing insert symmetric backward arrow
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if i != j:
                    if graph[i, j, tau] == '':
                        pass
                    elif graph[i, j, tau] == '-->':
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
    if verbosity_thesis > 3:
        print('len(ambiguous_locations)', len(ambiguous_locations))
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

        for graph_idx in (range(number_of_graph_combinations)):
            write_one_graph_combination_to_file(ambiguous_locations, links_permutations, graph_combinations, graph_idx)

        graph_combinations = load_all_graph_combinations_from_file(graph_combinations, number_of_graph_combinations)

        return graph_combinations


def load_all_graph_combinations_from_file(graph_combinations, number_of_graph_combinations):
    for graph_idx in range(number_of_graph_combinations):
        with open(checkpoint_path+'{}.pkl'.format(
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
            checkpoint_path+'{}.pkl'.format(
                graph_idx), 'wb') as f:
        pickle.dump(graph_combinations[graph_idx], f)
#
#
# def generate_symbolic_vars_dicts(var_names):
#     """
#     create symbolic external noise vars
#     one for each
#         - label
#         - external noise
#     """
#     symbolic_vars_dict = {}
#     symbolic_u_vars_dict = {}
#     plain_var_names_dict = {}
#     for i in var_names:
#         symbolic_u_vars_dict['u_' + str(i)] = sp.symbols('u_' + str(i))  # external noise
#         symbolic_vars_dict[str(i)] = 0  # sp.symbols(str(i))  # symbols(str(i))  # external noise symbols
#         plain_var_names_dict[str(i)] = sp.symbols(str(i))  # symbols(str(i))  # external noise symbols
#     return symbolic_vars_dict, symbolic_u_vars_dict, plain_var_names_dict
#
#
# def get_direct_influence_coeffs(
#         val_min,
#         graph,
#         var_names,
#         effect_label):
#     """
#     get_direct_influence_coeffs effect_label
#     input: val_min, graph, var_names, effect_label
#     output: direct_influence_coeffs
#     """
#     # get position of effect_label in ndarray var_names
#     effect_idx = np.where(np.array(var_names) == effect_label)[0][0]
#
#     direct_influence_coeffs = np.zeros(val_min.shape)
#     direct_influence_coeffs = direct_influence_coeffs[:, effect_idx, :]
#     graph_target = graph[:, effect_idx, :]
#     for time_lag in range(0, val_min.shape[2]):
#         for cause in range(len(graph_target)):
#             if graph_target[cause][time_lag] in [
#                 "-->",
#                 # "<--",
#                 # "<->",
#             ]:
#                 direct_influence_coeffs[cause][time_lag] = val_min[cause][effect_idx][time_lag]
#             elif graph_target[cause][time_lag] in [
#                 "---",
#                 "o--",
#                 "--o",
#                 "o-o",
#                 "o->",
#                 "x-o",
#                 "o-x",
#                 "x--",
#                 "--x",
#                 "x->",
#                 "x-x",
#                 "+->", ]:
#                 raise ValueError("invalid link type:" + str(graph_target[cause][time_lag]))
#             elif graph_target[cause][time_lag] in ['',
#                                                    "<--",
#                                                    "<->", ]:
#                 direct_influence_coeffs[cause][time_lag] = False
#             else:
#                 raise ValueError("unknown link type:" + str(graph_target[cause][time_lag]))
#     return direct_influence_coeffs
#
#
# def get_noise_value(symbolic_vars_dict, affected_var_label):
#     # get coeffs
#     coeffs = []
#     for coeff_and_symbol in symbolic_vars_dict[affected_var_label].args:
#
#         # if there is only one element in coeff_and_symbol, then it's already the coeff. add it then
#         if type(coeff_and_symbol).is_Float:
#             coeffs.append(coeff_and_symbol)
#
#         # if is_Mul, then it's coeff_and_symbol, and we need to get the coeff
#         elif type(coeff_and_symbol).is_Mul:
#             coeff = coeff_and_symbol.args[0]
#             # if datatype of i is float, then add it to coeffs
#             if type(coeff).is_Float:
#                 coeffs.append(coeff)
#             else:
#                 raise ValueError("did not get value:" + str(type(coeff)))
#
#     # make every coeff absolute
#     for i in range(len(coeffs)):
#         coeffs[i] = abs(coeffs[i])
#
#     # get noise coeff
#     noise_value = 1 - sum(coeffs)
#
#     if abs(noise_value) > 1:
#         print('Error: noise_value < 0 or noise_value > 1')
#         raise ValueError("noise_value < 0 or noise_value > 1")
#
#     return noise_value
#
#
# def fill_causes_of_one_affected_var(affected_var_label,
#                                     graph,
#                                     val_min,
#                                     var_names,
#                                     symbolic_vars_dict,
#                                     symbolic_u_vars_dict,
#                                     plain_var_names):
#     """
#     fill direct causes of an effect variable into symbolic_vars_dict
#     input: symbolic_vars_dict to modify, effect var, causes in form of val_min and graph
#     """
#     row_idx = -1  # row indicates which causing var
#     direct_influence_coeffs = get_direct_influence_coeffs(val_min, graph, var_names, affected_var_label)
#     for row_or_cause in direct_influence_coeffs:
#         row_idx += 1
#         col_idx = -1  # col indicates which time delay affected_var_label
#         for col_or_val in row_or_cause:
#             col_idx += 1
#
#             # get var name
#             cause_var_name = str(var_names[row_idx])
#
#             # get symbolic_cause_var_name
#             symbolic_cause_var_name = plain_var_names[cause_var_name]
#
#             # multiply coeff to symbolic_cause_var_name
#             symbol_val_to_add = symbolic_cause_var_name * col_or_val
#
#             # add symbol_val_to_add to dict
#             symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[
#                                                          affected_var_label] + symbol_val_to_add
#
#     # add noise term
#     noise_value = get_noise_value(symbolic_vars_dict, affected_var_label)  # get noise term
#     symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[affected_var_label] + noise_value * \
#                                              symbolic_u_vars_dict[
#                                                  'u_' + str(affected_var_label)]
#
#     return symbolic_vars_dict
#
#
# def chr_test(target_ans_per_graph_dict):
#     # # save str(target_ans_per_graph_dict) to file
#     # with open('target_ans_per_graph_dict_str_chr.txt', 'w') as f:
#     #     f.write(str(target_ans_per_graph_dict))
#
#     # load str(target_ans_per_graph_dict) from file
#     with open('target_ans_per_graph_dict_str_chr.txt', 'r') as f:
#         target_ans_per_graph_dict_gt = (f.read())
#
#     target_ans_per_graph_dict_str = str(target_ans_per_graph_dict)
#     same = target_ans_per_graph_dict_gt == target_ans_per_graph_dict_str
#     if same:
#         print('got expected result')
#         pass
#     else:
#         print('WARNING: target_ans_per_graph_dict is NOT the same')
#         ValueError('target_ans_per_graph_dict is not the same')
#
#
# def fill_target_ans_per_graph_dict(input_multiprocessing):
#     # unpack input_multiprocessing
#     graph_unambiguous, var_names, val_min, graph_combination_idx = input_multiprocessing
#
#     # plot_graph(val_min, graph_unambiguous, var_names, 'unambiguous graph no' +str(graph_combination_idx))
#
#     # ini symbolic vars dict
#     symbolic_vars_dict, symbolic_u_vars_dict, plain_var_names = generate_symbolic_vars_dicts(var_names)
#
#     # ini eq_dict
#     eq_list = []
#
#     # find causes of all variables
#     for var_name in var_names:
#         # fill causes of target var
#         symbolic_vars_dict = fill_causes_of_one_affected_var(affected_var_label=var_name,
#                                                              graph=graph_unambiguous,
#                                                              val_min=val_min,
#                                                              var_names=var_names,
#                                                              symbolic_vars_dict=symbolic_vars_dict,
#                                                              symbolic_u_vars_dict=symbolic_u_vars_dict,
#                                                              plain_var_names=plain_var_names)
#
#         # eq list: eq(long_equation, short_var_name)
#         eq_list.append(sp.Eq(symbolic_vars_dict[var_name], plain_var_names[var_name]))
#
#     # var list: [var_name1, var_name2, ..., noise_var_name1, noise_var_name2, ...]
#     var_list = []
#     # var names
#     for var_name in plain_var_names:
#         var_list.append(plain_var_names[var_name])
#     # noise names
#     for var_name in symbolic_u_vars_dict:
#         var_list.append(symbolic_u_vars_dict[var_name])
#
#     # solve(equations, symbols)
#     ans = sp.solve(eq_list, var_list)  # [target_label]
#
#     # store target result
#     # find target key
#     for i in range(len(list(ans.items()))):
#         if str(list(ans.items())[i][0]) == target_label:
#             target_ans_per_graph = list(ans.items())[i][1]
#     # test if target key was found by calling where it should be stored
#     try:
#         test = target_ans_per_graph
#     except KeyError:
#         ValueError('first item is not target_label')
#         print('valueerror: first item is not target_label')
#     return graph_combination_idx, target_ans_per_graph
#
#
# def compute_target_equations(val_min, graph, var_names):
#     """
#     compute target equations of all graph combinations
#     input: val_min, graph, var_names (loads from file)
#     output: target_equations_per_graph_dict
#     """
#     if verbosity_thesis > 0:
#         print('compute target equations ...')
#
#     # plot graph
#     plot_graph(val_min, graph, var_names, 'current graph estimate')
#
#     # drop redundant info in graph
#     graph = drop_redundant_information_due_to_symmetry(graph)
#
#     # find ambiguous link locations
#     ambiguous_locations = get_ambiguous_graph_locations(graph)
#
#     # create a list of all unique graph combinations
#     graph_combinations = create_all_graph_combinations(graph, ambiguous_locations)
#
#     # fill_target_ans_per_graph_dict
#     print('fill_target_ans_per_graph_dict ...')
#     # create input list
#     input_multiprocessing = []
#     for graph_combination_idx in range(len(graph_combinations)):
#         graph_combination = graph_combinations[graph_combination_idx]
#         input_multiprocessing.append((graph_combination, var_names, val_min, graph_combination_idx))
#
#     # multiprocessing
#     with Pool() as pool:
#         target_ans_per_graph_list = list(tqdm(pool.imap(fill_target_ans_per_graph_dict, input_multiprocessing)))
#     # target_ans_per_graph_list = []
#     # for input_multiprocessing_i in input_multiprocessing:
#     #     target_ans_per_graph_list.append(fill_target_ans_per_graph_dict(input_multiprocessing_i))
#
#
#     # results to dict
#     target_ans_per_graph_dict = {}
#     for i in range(len(target_ans_per_graph_list)):
#         target_ans_per_graph_dict[target_ans_per_graph_list[i][0]] = target_ans_per_graph_list[i][1]
#
#     print('... end fill_target_ans_per_graph_dict.')
#
#     # conduct test
#     # chr_test(target_ans_per_graph_dict)
#
#     # save target_ans_per_graph_dict and graph_combinations to file via pickle
#     with open(checkpoint_path+'target_eq_simulated.pkl', 'wb') as f:
#         pickle.dump(target_ans_per_graph_dict, f)
#     with open(checkpoint_path+'graph_combinations_simulated.pkl',
#               'wb') as f:
#         pickle.dump(graph_combinations, f)
#
#     return target_ans_per_graph_dict, graph_combinations
#
# # val_min, graph, var_names = load_results('chr')
# # var_names = [str(x) for x in var_names]
# # compute_target_equations(val_min, graph, var_names)
