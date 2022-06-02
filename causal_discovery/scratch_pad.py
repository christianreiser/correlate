import numpy as np
from sympy import symbols
from sympy.solvers import solve
from causal_discovery.LPCMCI.intervention import load_results
from sympy import Symbol
val_min, graph, var_names = load_results('chr')


"""def vars"""

# a = Symbol('a')
# b = Symbol('b')
# c = Symbol('c')
# d = Symbol('d')

symbolic_u_vars_dict = {}
symbolic_u_vars_dict['u_Mood'] = symbols('u_Mood')  # external noise symbols
symbolic_u_vars_dict['u_HP'] = symbols('u_HP')  # external noise symbols
symbolic_u_vars_dict['u_noise'] = symbols('u_noise')  # external noise symbols
symbolic_u_vars_dict['u_humid'] = symbols('u_humid')  # external noise symbols
# symbolic_u_vars_dict['u_steps'] = symbols('u_steps')  # external noise symbols

symbolic_vars_dict_new = {}
symbolic_vars_dict_new['Mood'] = symbols('Mood')  # external noise symbols
symbolic_vars_dict_new['HP'] = symbols('HP')  # external noise symbols
symbolic_vars_dict_new['noise'] = symbols('noise')  # external noise symbols
symbolic_vars_dict_new['humid'] = symbols('humid')  # external noise symbols
#symbolic_vars_dict['steps'] = symbols('steps')  # external noise symbols


converged = False
num_iterations = 0
while not converged:
    num_iterations += 1
    """copy symbolic_vars_dict to a new dict"""
    # old_a = a
    # old_b = b
    # old_c = c
    # old_d = d
    symbolic_vars_dict_old = symbolic_vars_dict_new.copy()

    """redefine"""
    # a = b
    # b = c
    # c = d
    # d = 1
    symbolic_vars_dict_new['Mood'] = 0.09*symbolic_vars_dict_new['humid']   + 0.91*symbolic_u_vars_dict['u_Mood'] # 0.08*symbolic_vars_dict['noise']
    symbolic_vars_dict_new['humid'] = 0.33*symbolic_vars_dict_new['noise'] + 0.66*symbolic_u_vars_dict['u_humid']
    symbolic_vars_dict_new['noise'] = 0.09*symbolic_vars_dict_new['HP'] + 0.91*symbolic_u_vars_dict['u_noise']
    symbolic_vars_dict_new['HP'] = 0.91*symbolic_u_vars_dict['u_HP'] + 0.09*symbolic_vars_dict_new['noise']
    #symbolic_vars_dict['steps'] = 1*symbolic_vars_dict['u_steps']

    # check if the new dict is the same as the old one
    if symbolic_vars_dict_new == symbolic_vars_dict_old and num_iterations < 30:
    # if a == old_a and b == old_b and c == old_c and d == old_d:
        converged = True
        print('same')
    else:
        converged = False
        print('different')
#
# symbolic_vars_dict['u_Mood'] = symbols('u_Mood')  # external noise symbols
# symbolic_vars_dict['u_HP'] = symbols('u_HP')  # external noise symbols
# symbolic_vars_dict['u_noise'] = symbols('u_noise')  # external noise symbols
# symbolic_vars_dict['u_humid'] = symbols('u_humid')  # external noise symbols
# symbolic_vars_dict['u_steps'] = symbols('u_steps')  # external noise symbols

# symbolic_vars_dict['Mood'] = 0.09*symbolic_vars_dict['HP'] - 0.08*symbolic_vars_dict['noise'] + 0.08*symbolic_vars_dict['u_Mood']
# symbolic_vars_dict['humid'] = 0.33*symbolic_vars_dict['noise'] - 0.66*symbolic_vars_dict['u_humid']
# symbolic_vars_dict['noise'] = 0.09*symbolic_vars_dict['HP'] + 0.9*symbolic_vars_dict['u_noise']
# symbolic_vars_dict['HP'] = 1*symbolic_vars_dict['u_HP'] + 0.9*symbolic_vars_dict['u_noise']
# symbolic_vars_dict['steps'] = 1*symbolic_vars_dict['u_steps']



# symbolic_vars_dict['Mood'] = 0.09*symbolic_vars_dict['HP'] - 0.08*symbolic_vars_dict['noise'] + 0.08*symbolic_vars_dict['u_Mood']
# symbolic_vars_dict['humid'] = 0.33*symbolic_vars_dict['noise'] - 0.66*symbolic_vars_dict['u_humid']
# symbolic_vars_dict['noise'] = 0.09*symbolic_vars_dict['HP'] + 0.9*symbolic_vars_dict['u_noise']
# symbolic_vars_dict['HP'] = 1*symbolic_vars_dict['u_HP'] + 0.9*symbolic_vars_dict['u_noise']
# symbolic_vars_dict['steps'] = 1*symbolic_vars_dict['u_steps']








print(solve(a,a))


print(solve(symbolic_vars_dict['Mood']-1, symbolic_vars_dict['Mood']))
# print(symbolic_vars_dict['noise'])