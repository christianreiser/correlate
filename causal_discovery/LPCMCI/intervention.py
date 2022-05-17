import numpy as np

def intervention():
    private_folder_path = '/home/chrei/code/quantifiedSelfData/'
    target_idx = 0
    target_label = 'Mood'  # label of interest

    # read val_min, graph and var_names to file via np.ndarray.tofile()
    val_min = np.load(str(private_folder_path) + 'val_min.npy', allow_pickle=True)
    graph = np.load(str(private_folder_path) + 'graph.npy', allow_pickle=True)
    var_names = np.load(str(private_folder_path) + 'var_names.npy', allow_pickle=True)
    # non_zero_inices = ['Mood', 'HumidInMax()', 'NoiseMax()', 'HeartPoints', 'Steps']

    direct_influence_coeffs = get_direct_influence_coeffs(val_min=val_min, graph=graph, var_names=var_names,
                                                          target_label=target_label)

    # todo search for intervention. below is hardcoded
    intervention_idx = 3
    print()

def get_direct_influence_coeffs(val_min, graph, var_names, target_label):
    # get position of target_label in ndarray var_names
    target_idx = np.where(var_names == target_label)[0][0]

    direct_influence_coeffs = np.zeros(val_min.shape)
    direct_influence_coeffs = direct_influence_coeffs[:, target_idx, :]
    graph_target = graph[:, target_idx, :]
    for time_lag in range(0, val_min.shape[2]):
        for cause in range(len(graph_target)):
            if graph_target[cause][time_lag] in [
                "---",
                "o--",
                "--o",
                "o-o",
                "o->",
                "-->",
                "<->",
                "x-o",
                "o-x",
                "x--",
                "--x",
                "x->",
                "x-x",
                "+->",
            ]:
                direct_influence_coeffs[cause][time_lag] = val_min[cause][target_idx][time_lag]
            else:
                direct_influence_coeffs[cause][time_lag] = False
    print()
    return direct_influence_coeffs


intervention()
print()
