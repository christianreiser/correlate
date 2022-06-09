import pickle

import numpy as np
import sympy as sp

from causal_discovery.LPCMCI.intervention import load_results, get_direct_influence_coeffs
from config import target_label
from tigramite import plotting as tp
from matplotlib import pyplot as plt


def drop_redundant_information_due_to_symmetry(graph):
    """
    drop redundant information due to symmetry
    """
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # drop all values of upper right triangle
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if i < j:
                    graph[i, j, tau] = ''
    return graph


def make_links_point_forward(graph):
    graph_new = np.copy(graph)
    # iterate through 3ed dimension (tau) of graph
    for tau in range(graph.shape[2]):
        # if value == '<--', then switch i and j and change value to '-->'
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i, j, tau] == '<--':
                    graph_new[i, j, tau] = ''
                    if graph[j, i, tau] != '':
                        raise ValueError('graph[j, i, tau] != ''')
                    graph_new[j, i, tau] = '-->'
    return graph_new


def get_ambiguous_graph_locations(graph):
    """
    get_ambiguous_graph_locations
    return:
    - [i, j, k, original_link, new_links]
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


def get_new_links_list(ambiguous_locations):
    """
    for each ambiguous location get the list of new possible links
    return: list of lists of new links
    input: ambiguous_locations
    """
    # of every list in ambiguous_locations, get 4th element (new_links) in a new list
    new_links_list = []  # [i, j, k, new_links]
    for ambiguous_location in ambiguous_locations:
        new_links_list.append([ambiguous_location[4]])
    return new_links_list


def get_links_permutations(new_links_list):
    """
    create new_links_list where each element is the list of a unique link combination
    links_permutations contains all links Permutations
    """
    links_permutations = []  # stupid ini
    for i in range(len(new_links_list) - 1):
        a = new_links_list[i][0]
        b = new_links_list[i + 1][0]
        links_permutations = [(x, y) for x in a for y in b]  # https://stackoverflow.com/a/39064769/7762867
        new_links_list[i + 1][0] = links_permutations

    if not links_permutations:
        links_permutations = np.transpose(np.asarray(new_links_list[0]))
    return links_permutations


def create_all_graph_combinations(graph, ambiguous_locations):
    """
    input: ambiguous_locations
    - [i, j, k, original_link, new_links]
    - e.g. [0, 1, 0, ['o-o'], [["-->", " <->", "<--"]]]
    """
    # create number_of_graph_combinations original graphs
    number_of_graph_combinations = get_number_of_graph_combinations(ambiguous_locations)

    # initialize graph_combinations
    graph_combinations = []
    for combi_idx in range(number_of_graph_combinations):
        graph_combinations.append(np.copy(graph))

    new_links_list = get_new_links_list(ambiguous_locations)
    links_permutations = get_links_permutations(new_links_list)

    # replace ambiguous links with unambiguous links
    for graph_idx in range(len(graph_combinations)):
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

            # replace graph_combinations[graph_idx][i, j, k] with new_link string
            graph_combinations[graph_idx][i, j, k] = old_link.replace(original_link, new_link)

        # make links point forward
        graph_combinations[graph_idx] = make_links_point_forward(graph_combinations[graph_idx])

    return graph_combinations


def generate_symbolic_vars_dicts(var_names):
    """
    create symbolic external noise vars
    one for each
        - label
        - external noise
    """
    symbolic_vars_dict = {}
    symbolic_u_vars_dict = {}
    plain_var_names_dict = {}
    for i in var_names:
        symbolic_u_vars_dict['u_' + str(i)] = sp.symbols('u_' + str(i))  # external noise
        symbolic_vars_dict[str(i)] = 0  # sp.symbols(str(i))  # symbols(str(i))  # external noise symbols
        plain_var_names_dict[str(i)] = sp.symbols(str(i))  # symbols(str(i))  # external noise symbols
    return symbolic_vars_dict, symbolic_u_vars_dict, plain_var_names_dict


def get_causes_of_one_affected_var(symbolic_vars_dict, affected_var_label):
    # get causes of affected var
    causes = symbolic_vars_dict[affected_var_label].free_symbols
    causes = [str(x) for x in causes]  # change causes to list with strings
    # for each item in causes remove everything behind '_tau='; e.g. 'x0_tau=0' -> 'x0'
    for i in range(len(causes)):
        causes[i] = causes[i][:causes[i].find('_tau=')]
    return causes


def get_noise_value(symbolic_vars_dict, affected_var_label):
    # get expression free symbols of symbolic_vars_dict[affected_var_label]
    coeff = symbolic_vars_dict[
        affected_var_label].expr_free_symbols  # expr_free_symbols is depricated but free_symbols doesn't contain the coeffs
    # sym to list
    coeff = [str(x) for x in coeff]
    # for all strings in sym, try to make string to float, otherwise drop it
    idx_to_drop = []
    for i in range(len(coeff)):
        try:
            coeff[i] = float(coeff[i])
        except:
            idx_to_drop.append(i)
    # revert idx_to_drop
    idx_to_drop.reverse()
    for i in idx_to_drop:
        del coeff[i]

    # make every coeff absolute
    for i in range(len(coeff)):
        coeff[i] = abs(coeff[i])

    # get noise coeff
    noise_value = 1 - sum(coeff)

    return noise_value


def fill_causes_of_one_affected_var(affected_var_label,
                                    graph,
                                    val_min,
                                    var_names,
                                    symbolic_vars_dict,
                                    symbolic_u_vars_dict,
                                    plain_var_names):
    """
    fill direct causes of a effect variable into symbolic_vars_dict
    input: symbolic_vars_dict to modify, effect var, causes in form of val_min and graph
    """
    row_idx = -1  # row indicates which causing var
    direct_influence_coeffs = get_direct_influence_coeffs(val_min, graph, var_names, affected_var_label)
    for row_or_cause in direct_influence_coeffs:
        row_idx += 1
        col_idx = -1  # col indicates which time delay affected_var_label
        for col_or_val in row_or_cause:
            col_idx += 1

            # get var name
            cause_var_name = var_names[row_idx]

            # get symbolic_cause_var_name
            symbolic_cause_var_name = plain_var_names[cause_var_name]

            # multiply coeff to symbolic_cause_var_name
            symbol_val_to_add = symbolic_cause_var_name * col_or_val

            # add symbol_val_to_add to dict
            symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[
                                                         affected_var_label] + symbol_val_to_add

    # add noise term
    noise_value = get_noise_value(symbolic_vars_dict, affected_var_label)  # get noise term
    symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[affected_var_label] + noise_value * \
                                             symbolic_u_vars_dict[
                                                 'u_' + str(affected_var_label)]

    return symbolic_vars_dict


def plot_graph(val_min, graph, var_names):
    tp.plot_graph(
        val_matrix=val_min,
        link_matrix=graph,
        var_names=var_names,
        # link_colorbar_label='cross-MCI',
        # node_colorbar_label='auto-MCI',
        figsize=(10, 6),
    )
    plt.show()

    # Plot time series graph
    tp.plot_time_series_graph(
        figsize=(12, 8),
        val_matrix=val_min,
        link_matrix=graph,
        var_names=var_names,
        link_colorbar_label='MCI',
    )
    plt.show()


def chr_test(target_ans_per_graph_dict):
    # # save str(target_ans_per_graph_dict) to file
    # with open('target_ans_per_graph_dict_str_chr.txt', 'w') as f:
    #     f.write(str(target_ans_per_graph_dict))

    # load str(target_ans_per_graph_dict) from file
    with open('../causal_discovery/target_ans_per_graph_dict_str_chr.txt', 'r') as f:
        target_ans_per_graph_dict_gt = (f.read())

    target_ans_per_graph_dict_str = str(target_ans_per_graph_dict)
    same = target_ans_per_graph_dict_gt == target_ans_per_graph_dict_str
    if same:
        print('got expected result')
        pass
    else:
        print('WARNING: target_ans_per_graph_dict is NOT the same')
        ValueError('target_ans_per_graph_dict is not the same')


def compute_target_equations():
    """
    compute target equations of all graph combinations
    input: val_min, graph, var_names (loads from file)
    output: target_equations_per_graph_dict
    """
    # load data
    val_min, graph, var_names = load_results('chr')

    # plot graph
    # plot_graph(val_min, graph, var_names)

    # drop redundant info in graph
    graph = drop_redundant_information_due_to_symmetry(graph)

    # find ambiguous link locations
    ambiguous_locations = get_ambiguous_graph_locations(graph)

    # create a list of all unique graph combinations
    graph_combinations = create_all_graph_combinations(graph, ambiguous_locations)

    # ini result dict
    target_ans_per_graph_dict = {}

    # ini idx
    graph_idx = -1
    # for all graph combinations
    for graph_unambiguous in graph_combinations:
        graph_idx += 1

        # ini symbolic vars dict
        symbolic_vars_dict, symbolic_u_vars_dict, plain_var_names = generate_symbolic_vars_dicts(var_names)

        # ini eq_dict
        eq_list = []

        # find causes of all variables
        for var_name in var_names:
            # fill causes of target var
            symbolic_vars_dict = fill_causes_of_one_affected_var(affected_var_label=var_name,
                                                                 graph=graph_unambiguous,
                                                                 val_min=val_min,
                                                                 var_names=var_names,
                                                                 symbolic_vars_dict=symbolic_vars_dict,
                                                                 symbolic_u_vars_dict=symbolic_u_vars_dict,
                                                                 plain_var_names=plain_var_names)

            # eq list: eq(long_equation, short_var_name)
            eq_list.append(sp.Eq(symbolic_vars_dict[var_name], plain_var_names[var_name]))

        # var list: [var_name1, var_name2, ..., noise_var_name1, noise_var_name2, ...]
        var_list = []
        # var names
        for var_name in plain_var_names:
            var_list.append(plain_var_names[var_name])
        # noise names
        for var_name in symbolic_u_vars_dict:
            var_list.append(symbolic_u_vars_dict[var_name])

        # solve(equations, symbols)
        ans = sp.solve(eq_list, var_list)  # [target_label]

        # store target result
        # check if first is target result
        if str(list(ans.items())[0][0]) == target_label:
            target_ans_per_graph_dict[graph_idx] = list(ans.items())[0][1]
        else:
            ValueError('first item is not target_label')

    # conduct test
    chr_test(target_ans_per_graph_dict)

    # save target_ans_per_graph_dict to file via pickle
    with open('../causal_discovery/target_eq_chr.pkl', 'wb') as f:
        pickle.dump(target_ans_per_graph_dict, f)

    return target_ans_per_graph_dict
# compute_target_equations()