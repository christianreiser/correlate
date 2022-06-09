import pickle

import numpy as np


def drop_unintervenable_variables():
    """
    drop variables from equations which can't be intervened upon
    """
    # load target_ans_per_graph_dict from file via pickle
    with open('target_eq_chr.pkl', 'rb') as f:
        target_eq = pickle.load(f)

        # names of unintervenable vars
        unintervenable_vars = ['u_Mood']

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
                if var_name_in_eq in unintervenable_vars:
                    # if unintervenable, drop var name from eq
                    target_eq[eq_idx] = eq.subs(var_name_in_eq, 0)
    return target_eq


def find_most_optimistic_intervention(target_eq):
    """
    find variable name with the largest absolute coefficient in target_eq
    input: target_eq
    output: most_optimistic_intervention
    """
    # get var names
    var_names = [str(var_name) for var_name in target_eq[0].free_symbols]

    # get coefficients
    coeffs = [target_eq[0].coeff(var_name) for var_name in var_names]

    # find most optimistic intervention
    most_optimistic_intervention = var_names[np.argmax(np.abs(coeffs))]

    # find value of most optimistic intervention
    # most_optimistic_intervention_value = np.max(np.abs(coeffs))

    return most_optimistic_intervention


target_eq = drop_unintervenable_variables()

proposed_intervention = find_most_optimistic_intervention(target_eq)
print()
