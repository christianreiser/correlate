import numpy as np
from sympy import symbols

from causal_discovery.LPCMCI.intervention import load_results

val_min, graph, var_names = load_results('chr')


symbolic_vars_dict = {}
for i in var_names:
    # symbolic_vars_dict['u_' + str(i)] = symbols('u_' + str(i))  # todo external noise symbols
    # create symbolic vars for each tau
    # for delay in range(tau_max):
    #     symbolic_vars_dict[str(i) + '_tau=' + str(delay)] = symbols('u_' + str(i) + '_tau=' + str(delay))
    symbolic_vars_dict[str(i)] = symbols(str(i))  # external noise symbols

affected_var_label = 'Mood'
symbolic_vars_dict[affected_var_label] = symbolic_vars_dict[affected_var_label] + symbolic_vars_dict[affected_var_label] * 0.22 - 1* symbolic_vars_dict[affected_var_label]

print()