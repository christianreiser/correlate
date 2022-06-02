import numpy as np
from sympy import symbols
# x0 + 2*x1 = 1 and 3*x0 + 5*x1 = 2:
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
    return graph


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
    number_of_graph_combinations = 0
    for ambiguous_location in ambiguous_locations:
        number_of_graph_combinations = number_of_graph_combinations + len(ambiguous_location[4])
    return number_of_graph_combinations


def get_new_links_list(ambiguous_locations):
    # of every list in ambiguous_locations, get 4th element (new_links) in a new list
    new_links_list = []  # [i, j, k, new_links]
    for ambiguous_location in ambiguous_locations:
        new_links_list.append([ambiguous_location[4]])
    # flatten new_links_list    new_links_list_flat = [item for sublist in new_links_list for item in sublist]
    new_links_list_flat = [item for sublist in new_links_list for item in sublist][0]
    return new_links_list_flat


def create_all_graph_combinations(graph, ambiguous_locations):
    """
    input: ambiguous_locations
    - [i, j, k, original_link, new_links]
    - e.g. [0, 1, 0, ['o-o'], [["-->", " <->", "<--"]]]
    """
    # initialize graph_combinations
    # create number_of_graph_combinations original graphs
    number_of_graph_combinations = get_number_of_graph_combinations(ambiguous_locations)
    graph_combinations = []
    for combi_idx in range(number_of_graph_combinations):
        graph_combinations.append(np.copy(graph))

    new_links_list_flat = get_new_links_list(ambiguous_locations)

    # replace ambiguous links with unambiguous links
    for graph_idx in range(len(graph_combinations)):
        for ambiguous_location in ambiguous_locations:
            # get original link
            original_link = ambiguous_location[3]
            # get new links
            new_links = ambiguous_location[4]
            # get i, j, k
            i = ambiguous_location[0]
            j = ambiguous_location[1]
            k = ambiguous_location[2]
            new_link = new_links_list_flat[graph_idx]
            # replace original link with new link
            old_string = graph_combinations[graph_idx][i, j, k]
            # replace graph_combinations[graph_idx][i, j, k] with new_link
            graph_combinations[graph_idx][i, j, k] = old_string.replace(original_link, new_link)
    return graph_combinations


def generate_symbolic_vars_dict(var_names, tau_max):
    """
    create symbolic external noise vars
    one for each
        - label
        - *delay
        - external noise
    """
    symbolic_vars_dict = {}
    symbolic_u_vars_dict = {}
    for i in var_names:
        symbolic_u_vars_dict['u_' + str(i)] = symbols('u_' + str(i)) # external noise
        symbolic_vars_dict[str(i)] = symbols(str(i))  # external noise symbols
    return symbolic_vars_dict, symbolic_u_vars_dict


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
    coeff = symbolic_vars_dict[affected_var_label].expr_free_symbols # expr_free_symbols is depricated but free_symbols doesn't contain the coeffs
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


def fill_causes_of_one_affected_var(affected_var_label, graph, val_min, var_names, symbolic_vars_dict, symbolic_u_vars_dict):
    """
    fill direct causes of a effect variable into symbolic_vars_dict
    input: symbolic_vars_dict to modify, effect var, causes in form of val_min and graph
    """
    row_idx = -1  # row indicates which causing var
    for row_or_cause in get_direct_influence_coeffs(val_min, graph, var_names, affected_var_label):
        row_idx += 1
        col_idx = -1  # col indicates which time delay affected_var_label
        for col_or_val in row_or_cause:
            col_idx += 1
            cause_var_name = var_names[row_idx]
            symbolic_cause_var_name = symbolic_vars_dict[cause_var_name]

            symbol_val_to_add = symbolic_cause_var_name * col_or_val
            symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[
                                                         affected_var_label] + symbol_val_to_add
            # todo check if below is good
            # problem is that otherwise mood is too big
            if str(symbolic_cause_var_name.free_symbols)[1:-1] == affected_var_label:
                workaround_adjustment = 1
            else:
                workaround_adjustment = 0
            symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[
                                                         affected_var_label] - workaround_adjustment * symbolic_cause_var_name

    # add noise term
    noise_value = get_noise_value(symbolic_vars_dict, affected_var_label)  # get noise term
    symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[affected_var_label] + noise_value * symbolic_u_vars_dict[
        'u_' + str(affected_var_label)]



    return symbolic_vars_dict


def main():
    # load results
    val_min, graph, var_names = load_results('chr')

    graph = drop_redundant_information_due_to_symmetry(graph)

    ambiguous_locations = get_ambiguous_graph_locations(graph)

    graph_combinations = create_all_graph_combinations(graph, ambiguous_locations)

    # do for all graph combinations
    for graph_unambiguous in graph_combinations:
        graph_unambiguous = make_links_point_forward(graph_unambiguous)

        # ini symbolic vars dict
        symbolic_vars_dict, symbolic_u_vars_dict = generate_symbolic_vars_dict(var_names, graph_unambiguous.shape[2])

        # iterate until convergence
        converged = False
        num_iterations = 0
        while not converged:
            num_iterations += 1
            """copy symbolic_vars_dict to a new dict"""
            symbolic_vars_dict_old = symbolic_vars_dict.copy()


            # find causes of all variables
            for var_name in var_names:
                # fill causes of target var
                symbolic_vars_dict = fill_causes_of_one_affected_var(affected_var_label=var_name,
                                                                     graph=graph_unambiguous,
                                                                     val_min=val_min,
                                                                     var_names=var_names,
                                                                     symbolic_vars_dict=symbolic_vars_dict,
                                                                     symbolic_u_vars_dict=symbolic_u_vars_dict)
            # check if converged
            if symbolic_vars_dict == symbolic_vars_dict_old and num_iterations < 30:
                # if a == old_a and b == old_b and c == old_c and d == old_d:
                converged = True
                print('same')
            else:
                converged = False
                print('different')



        print(symbolic_vars_dict['Mood'])
        print(symbolic_vars_dict['HeartPoints'])
        print()


main()
