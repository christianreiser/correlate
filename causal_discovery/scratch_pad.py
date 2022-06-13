import numpy as np

var_names= [0, 1, 2, 3, 4, 5, 6, 7]
effect_label = 0

# get index of var_names where item is zero
# non_zero_indices = np.where(df[var_names] != 0)[0]
names = np.array(np.array(var_names))
ans = np.where(names == effect_label)
print()
print()