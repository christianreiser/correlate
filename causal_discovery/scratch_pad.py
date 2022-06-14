# # save symbolic_vars_dict with pickle to file
# with open(
#         '/home/chrei/PycharmProjects/correlate/intervention_proposal/symbolic_vars_dicts/tmp.pkl', 'wb') as f:
#     pickle.dump(symbolic_vars_dict, f)

# load symbolic_vars_dict from file via pickle
import pickle

with open(
        '/home/chrei/PycharmProjects/correlate/tmp.pkl', 'rb') as f:
    symbolic_vars_dict = pickle.load(f)

coeff_and_symbols = symbolic_vars_dict[
    '0'].expr_free_symbols
coeffs = []
for i in coeff_and_symbols:
    # if datatype of i is float, then add it to coeffs
    if type(i).is_Float:
        coeffs.append(i)

print()
