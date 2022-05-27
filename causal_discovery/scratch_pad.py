import numpy as np
from sympy import symbols
# x0 + 2*x1 = 1 and 3*x0 + 5*x1 = 2:
from causal_discovery.LPCMCI.intervention import load_results, get_direct_influence_coeffs
from config import target_label


def generate_symbolic_vars_dict(var_names, tau_max):
    """
    create symbolic external noise vars
    one for each
        - label
        - *delay
        - external noise
    """
    symbolic_vars_dict = {}
    for i in var_names:
        symbolic_vars_dict['u_' + str(i)] = symbols('u_' + str(i))  # external noise symbols
        # create symbolic vars for each tau
        for delay in range(tau_max):
            symbolic_vars_dict[str(i) + '_tau=' + str(delay)] = symbols('u_' + str(i) + '_tau=' + str(delay))
    return symbolic_vars_dict


def get_causes_of_one_affected_var(symbolic_vars_dict, affected_var_label):
    # get causes of affected var
    causes = symbolic_vars_dict[affected_var_label].free_symbols
    causes = [str(x) for x in causes]  # change causes to list with strings
    # for each item in causes remove everything behind '_tau='; e.g. 'x0_tau=0' -> 'x0'
    for i in range(len(causes)):
        causes[i] = causes[i][:causes[i].find('_tau=')]
    return causes


def fill_causes_of_one_affected_var(affected_var_label, graph, val_min, var_names, symbolic_vars_dict,
                                    causes_identified):
    """
    fill direct causes given effect variable into symbolic_vars_dict
    input: symbolic_vars_dict to modify, effect var, causes in form of val_min and graph
    """
    affected_var_label_tau0 = affected_var_label + '_tau=0'
    row_idx = -1  # row indicates which causing var
    for row_or_cause in get_direct_influence_coeffs(val_min, graph, var_names, affected_var_label):
        row_idx += 1
        col_idx = -1  # col indicates which time delay
        for col_or_val in row_or_cause:
            col_idx += 1
            cause_var_name = var_names[row_idx] + '_tau=' + str(col_idx)
            symbolic_cause_var_name = symbolic_vars_dict[cause_var_name]
            symbolic_vars_dict[affected_var_label_tau0] = symbolic_vars_dict[
                                                              affected_var_label_tau0] + symbolic_cause_var_name * col_or_val
    # mark affected var as identified
    causes_identified[affected_var_label] = True

    causes = get_causes_of_one_affected_var(symbolic_vars_dict, affected_var_label)

    return symbolic_vars_dict, causes_identified, causes


def main():
    # load results
    val_min, graph, var_names = load_results('chr')

    # ini causes identified dict
    causes_identified = {}
    for var_name in var_names:
        causes_identified[var_name] = False

    # ini symbolic vars dict
    symbolic_vars_dict = generate_symbolic_vars_dict(var_names, graph.shape[2])

    # fill causes of target var
    symbolic_vars_dict, causes_identified, causes_target = fill_causes_of_one_affected_var(affected_var_label=target_label,
                                                                                    graph=graph,
                                                                                    val_min=val_min,
                                                                                    var_names=var_names,
                                                                                    symbolic_vars_dict=symbolic_vars_dict,
                                                                                    causes_identified=causes_identified)

    # for cause not identified yet
    for cause in causes_target:
        if not causes_identified[cause]:
            symbolic_vars_dict, causes_identified, causes_new = fill_causes_of_one_affected_var(
                affected_var_label=cause, graph=graph, val_min=val_min, var_names=var_names,
                symbolic_vars_dict=symbolic_vars_dict, causes_identified=causes_identified)
    print()

main()
