import numpy as np
from config import private_folder_path, target_label

# function that loads val_min, graph, and var_names from a file and allow_pickle=True
def load_results(name_extension):
    val_min = np.load(str(private_folder_path) + 'val_min_'+str(name_extension)+'.npy', allow_pickle=True)
    graph = np.load(str(private_folder_path) + 'graph_'+str(name_extension)+'.npy', allow_pickle=True)
    var_names = np.load(str(private_folder_path) + 'var_names_'+str(name_extension)+'.npy', allow_pickle=True)
    return val_min, graph, var_names


def intervention():
    # load results
    val_min, graph, var_names = load_results('chr')
    direct_influence_coeffs = get_direct_influence_coeffs(val_min=val_min, graph=graph, var_names=var_names, effect_label=target_label)

    # todo search for intervention. below is hardcoded
    intervention_idx = 3
    print()

def get_direct_influence_coeffs(
        val_min,
        graph,
        var_names,
        effect_label):
    """
    get_direct_influence_coeffs effect_label
    input: val_min, graph, var_names, effect_label
    output: direct_influence_coeffs
    """
    # get position of effect_label in ndarray var_names
    effect_idx = np.where(var_names == effect_label)[0][0]

    direct_influence_coeffs = np.zeros(val_min.shape)
    direct_influence_coeffs = direct_influence_coeffs[:, effect_idx, :]
    graph_target = graph[:, effect_idx, :]
    for time_lag in range(0, val_min.shape[2]):
        for cause in range(len(graph_target)):
            if graph_target[cause][time_lag] in [
                "-->",
                # "<--",
                # "<->",
            ]:
                direct_influence_coeffs[cause][time_lag] = val_min[cause][effect_idx][time_lag]
            elif graph_target[cause][time_lag] in [
                "---",
                "o--",
                "--o",
                "o-o",
                "o->",
                "x-o",
                "o-x",
                "x--",
                "--x",
                "x->",
                "x-x",
                "+->",]:
                    raise ValueError("invalid link type:" + str(graph_target[cause][time_lag]))
            elif graph_target[cause][time_lag] in ['',
                                                   "<--",
                                                   "<->", ]:
                direct_influence_coeffs[cause][time_lag] = False
            else:
                raise ValueError("unknown link type:" + str(graph_target[cause][time_lag]))

    print()
    return direct_influence_coeffs


intervention()
print()
