import numpy as np

from config import target_label, verbosity_thesis, n_vars_all, frac_latents
from config_helper import get_measured_labels


def drop_unintervenable_variables(target_eq, random_state):
    """
    drop variables from equations which can't be intervened upon
    """


    # names of unintervenable vars
    # targetlabel
    unintervenable_vars = ['u_'+str(target_label)]
    # measured vars
    measured_labels, measured_label_to_idx, unmeasured_labels_strs = get_measured_labels(n_vars_all, random_state, frac_latents)
    for measured_label_idx in range(len(measured_labels)):
        measured_labels[measured_label_idx] = 'u_' + measured_labels[measured_label_idx]

    # loop through equations
    for eq_idx in range(len(target_eq)):
        eq = target_eq[eq_idx]

        # get var names
        var_names_in_eq = eq.free_symbols

        # make var_names_in_eq a list of strings
        var_names_in_eq = [str(var_name) for var_name in var_names_in_eq]

        # loop through var names in eq
        for var_name_in_eq in var_names_in_eq:

            # check if var_name_in_eq is unintervenable
            if var_name_in_eq not in measured_labels: # todo remove after check
                print('todo check')

            if var_name_in_eq in unintervenable_vars or var_name_in_eq not in measured_labels:
                # if unintervenable, drop var name from eq
                target_eq[eq_idx] = eq.subs(var_name_in_eq, 0)
    return target_eq


def find_most_optimistic_intervention(target_eqs):
    """
    find variable name with the largest absolute coefficient in target_eq
    input: target_eqs
    output: most_optimistic_intervention
    """
    largest_abs_coeff = 0
    largest_coeff = 0
    best_intervention_var_name = None
    most_optimistic_graph_idx = None

    for equation_idx in range(len(target_eqs)):

        # get var names
        var_names = [str(var_name) for var_name in target_eqs[equation_idx].free_symbols]

        # get coefficients
        coeffs = [target_eqs[0].coeff(var_name) for var_name in var_names]

        # get most extreme coeff
        abs_coeffs = [np.abs(coeff) for coeff in coeffs]
        if len(abs_coeffs) > 0:
            largest_abs_coeff_in_one_graph = np.max(abs_coeffs)
            largest_coeff_in_one_graph = np.max(coeffs)

            # if better coeff is found
            if np.abs(largest_abs_coeff) < np.abs(largest_abs_coeff_in_one_graph):

                # update value of most optimistic intervention
                largest_abs_coeff = largest_abs_coeff_in_one_graph
                largest_coeff = largest_coeff_in_one_graph

                # update most optimistic intervention
                best_intervention_var_name = var_names[np.argmax(np.abs(coeffs))]

                most_optimistic_graph_idx = equation_idx


    if verbosity_thesis >0:
        print('largest_abs_coeff: ' + str(largest_abs_coeff))
        print('best_intervention: ' + str(best_intervention_var_name))
        print('most_optimistic_graph_idx: ' + str(most_optimistic_graph_idx))
    return largest_abs_coeff, best_intervention_var_name, most_optimistic_graph_idx, largest_coeff

# # load target_ans_per_graph_dict from file via pickle
# with open(checkpoint_path+'target_eq_chr.pkl', 'rb') as f:
#     target_eq = pickle.load(f)
#
# target_eq = drop_unintervenable_variables(target_eq, random_state)
#
# largest_abs_coeff, best_intervention, most_optimistic_graph_idx = find_most_optimistic_intervention(target_eq)
# print()
