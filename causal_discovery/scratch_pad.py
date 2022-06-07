import numpy as np
import sympy as sp
from sympy.solvers import solve
from causal_discovery.LPCMCI.intervention import load_results

val_min, graph, var_names = load_results('chr')

"""def vars"""

# a = Symbol('a')
# b = Symbol('b')
# c = Symbol('c')
# d = Symbol('d')

symbolic_u_vars_dict = {}
symbolic_u_vars_dict['u_Mood'] = sp.symbols('u_Mood')  # external noise symbols
symbolic_u_vars_dict['u_HP'] = sp.symbols('u_HP')  # external noise symbols
symbolic_u_vars_dict['u_noise'] = sp.symbols('u_noise')  # external noise symbols
symbolic_u_vars_dict['u_humid'] = sp.symbols('u_humid')  # external noise symbols

symbolic_vars_dict_new = {}
symbolic_vars_dict_new['Mood'] = sp.symbols('Mood')  # external noise symbols
symbolic_vars_dict_new['HP'] = sp.symbols('HP')  # external noise symbols
symbolic_vars_dict_new['noise'] = sp.symbols('noise')  # external noise symbols
symbolic_vars_dict_new['humid'] = sp.symbols('humid')  # external noise symbols

# symbolic_vars_dict_new['Mood'] = 0.09 * symbolic_vars_dict_new['humid'] + 0.91 * symbolic_u_vars_dict[
#     'u_Mood']  # 0.08*symbolic_vars_dict['noise']
# symbolic_vars_dict_new['humid'] = 0.33 * symbolic_vars_dict_new['noise'] + 0.66 * symbolic_u_vars_dict['u_humid']
# symbolic_vars_dict_new['noise'] = 0.09 * symbolic_vars_dict_new['HP'] + 0.91 * symbolic_u_vars_dict['u_noise']
# symbolic_vars_dict_new['HP'] = 1 * symbolic_u_vars_dict['u_HP']  # + 0.09*symbolic_vars_dict_new['noise']

# x, y, z = sp.symbols('x, y, z')
# eq1 = sp.Eq(x + y + z, 1)             # x + y + z  = 1
eqmood = sp.Eq(0.1 * symbolic_vars_dict_new['humid'] + 0.9 * symbolic_u_vars_dict[
    'u_Mood'], symbolic_vars_dict_new['Mood'])
eqhumid = sp.Eq(0.3 * symbolic_vars_dict_new['noise'] + 0.7 * symbolic_u_vars_dict['u_humid'],
                symbolic_vars_dict_new['humid'])  # x + y + 2z = 3
eqnoise = sp.Eq(0.2 * symbolic_vars_dict_new['HP'] + 0.8 * symbolic_u_vars_dict['u_noise'],
                symbolic_vars_dict_new['noise'])  # x + y + 2z = 3
eqhp = sp.Eq(1 * symbolic_u_vars_dict['u_HP'], symbolic_vars_dict_new['HP'])  # x + y + 2z = 3
ans = sp.solve((eqmood, eqhumid, eqnoise, eqhp), (symbolic_vars_dict_new['Mood'], symbolic_vars_dict_new['humid'],
    symbolic_vars_dict_new['noise'], symbolic_vars_dict_new['HP'],
    symbolic_u_vars_dict['u_Mood'], symbolic_u_vars_dict['u_HP'], symbolic_u_vars_dict['u_noise'],
    symbolic_u_vars_dict['u_humid']))

print(ans)
# converged = False
# num_iterations = 0
# while not converged:
#     num_iterations += 1
#     """copy symbolic_vars_dict to a new dict"""
#     # old_a = a
#     # old_b = b
#     # old_c = c
#     # old_d = d
#     symbolic_vars_dict_old = symbolic_vars_dict_new.copy()
#
#     """redefine"""
#     # a = b
#     # b = c
#     # c = d
#     # d = 1
#     symbolic_vars_dict_new['Mood'] = 0.09*symbolic_vars_dict_new['humid']   + 0.91*symbolic_u_vars_dict['u_Mood'] # 0.08*symbolic_vars_dict['noise']
#     symbolic_vars_dict_new['humid'] = 0.33*symbolic_vars_dict_new['noise'] + 0.66*symbolic_u_vars_dict['u_humid']
#     symbolic_vars_dict_new['noise'] = 0.09*symbolic_vars_dict_new['HP'] + 0.91*symbolic_u_vars_dict['u_noise']
#     symbolic_vars_dict_new['HP'] = 1*symbolic_u_vars_dict['u_HP'] # + 0.09*symbolic_vars_dict_new['noise']
#
#     # check if the new dict is the same as the old one
#     if symbolic_vars_dict_new == symbolic_vars_dict_old and num_iterations < 30:
#     # if a == old_a and b == old_b and c == old_c and d == old_d:
#         converged = True
#         print('same')
#     else:
#         converged = False
#         print('different')
#


print(solve(a, a))

print(solve(symbolic_vars_dict['Mood'] - 1, symbolic_vars_dict['Mood']))
# print(symbolic_vars_dict['noise'])
