import numpy as np
from tigramite import plotting as tp
from matplotlib import pyplot, pyplot as plt

my_dict = {0: [
    ((0, -1), 0.85, 'removeme'),
    ((1, 0), -0.5, 'removeme'),
    ((2, -1), 0.7, 'removeme'),
],
    1: [((1, -1), 0.8, 'removeme'),
        ((2, 0), 0.7, 'removeme')
        ],
    2: [((2, -1), 0.9, 'removeme')
        ],
    3: [((3, -1), 0.8, 'removeme'),
        ((0, -2), 0.4, 'removeme')
        ],
}

len_dict = len(my_dict)
max_time_lag = 0

for key in my_dict:
    my_list = my_dict[key]
    len_my_list = len(my_list)
    modified_list = []
    for list_index in range(len_my_list):
        my_tuple = my_list[list_index]
        modified_tuple = my_tuple[:-1]
        modified_list.append(modified_tuple)

        # get max time lag
        if max_time_lag > modified_tuple[0][1]:
            max_time_lag = modified_tuple[0][1]

    my_dict.update({key: modified_list})

max_time_lag = - max_time_lag

graph = np.ndarray(shape=(len_dict, len_dict, max_time_lag + 1), dtype='U3')
val = np.zeros(shape=(len_dict, len_dict, max_time_lag + 1), dtype=float)
print(graph)
for key in my_dict:
    my_list = my_dict[key]
    len_my_list = len(my_list)
    for list_index in range(len_my_list):
        my_tuple = my_list[list_index]
        effected_index = key
        cause_index = my_tuple[0][0]
        lag = -my_tuple[0][1]
        link_strength = my_tuple[1]
        graph[cause_index][effected_index][lag] = '-->'
        if lag == 0:
            graph[effected_index][cause_index][lag] = '<--'
        val[effected_index][cause_index][lag] = link_strength
        val[cause_index][effected_index][lag] = link_strength
print('\n\n\n\n', graph)

# plot original DAG
tp.plot_graph(
    val_matrix=val,  # original_vals
    link_matrix=graph,
    var_names=range(len_dict),
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    figsize=(10, 6),
)

# Plot time series graph
tp.plot_time_series_graph(
    figsize=(12, 8),
    val_matrix=val,  # original_vals None
    link_matrix=graph,
    var_names=range(len_dict),
    link_colorbar_label='MCI',
)
plt.show()